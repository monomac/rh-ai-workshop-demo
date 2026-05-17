"""Feast IO helpers — shared by notebook 02 (train), notebook 03 (score),
and the KFP pipeline.

Why this lives here, not in the notebook:
- Keeps the `feast apply` / materialize / historical-join boilerplate in one
  place so the notebooks stay readable.
- Same code path runs in the pipeline component, eliminating train-pipeline
  skew in feature computation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd

from invoice_features import FEATURE_COLUMNS, build_features

# Where the feast project lives in this repo.
FEAST_REPO_PATH = Path(__file__).parent / "feast_repo"

# Where Feast's FileSource reads from. Lives next to the feast_repo so the
# Dask offline engine can find it without s3fs. data_sources.py uses the
# matching relative path "data/invoice_features.parquet".
LOCAL_PARQUET_PATH = FEAST_REPO_PATH / "data" / "invoice_features.parquet"

# Same parquet ALSO uploaded to MinIO for audit + cross-environment access.
# Notebook 02 writes both copies; only the local one is read by Feast.
FEAST_PARQUET_KEY = "feast/invoice_features.parquet"


# ---------------------------------------------------------------------------
# Build the feature parquet that the FileSource reads
# ---------------------------------------------------------------------------


def write_feature_parquet(
    invoices_df: pd.DataFrame,
    *,
    artifact: Optional[dict] = None,
    s3_client=None,
    bucket: Optional[str] = None,
    local_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute features for every invoice and write a parquet for Feast.

    The parquet has one row per `invoice_id` with all nine feature columns,
    plus `event_timestamp` (= issued_on at midnight UTC). Feast's
    FileSource is configured to point at this object.

    Parameters
    ----------
    invoices_df : raw invoice dataframe (output of `load_invoices`)
    artifact    : if given, must contain `category_index`,
                  `vendor_freq_table`, `category_stats` — features will be
                  built using the train-time lookups for parity with score.
                  If None, lookups are derived from `invoices_df` (training
                  mode).
    s3_client   : boto3 S3 client. If given, the parquet is uploaded to
                  `bucket`/feast/invoice_features.parquet.
    bucket      : MinIO bucket name (required when s3_client is given).
    local_path  : where the parquet is written for Feast's offline reads.
                  Defaults to `LOCAL_PARQUET_PATH` (next to feast_repo);
                  Feast's FileSource expects to find it there.

    Returns
    -------
    (feature_df, meta) : feature dataframe (also written) plus the meta
                         dict from `build_features` so the caller can save
                         the lookup tables alongside the model.
    """
    if artifact is None:
        X, meta = build_features(invoices_df)
    else:
        X, meta = build_features(
            invoices_df,
            category_index=artifact["category_index"],
            vendor_freq_table=pd.Series(artifact["vendor_freq_table"]),
            category_stats=pd.DataFrame.from_dict(
                artifact["category_stats"], orient="index"
            ),
        )

    feat = pd.DataFrame(X, columns=FEATURE_COLUMNS).astype("float64")
    feat["invoice_id"] = invoices_df["invoice_id"].astype(str).values
    # Feast expects microsecond precision (timestamp[us]); pandas defaults to
    # nanosecond which trips materialize-incremental schema-validation.
    feat["event_timestamp"] = (
        pd.to_datetime(invoices_df["issued_on"]).dt.tz_localize("UTC")
          .astype("datetime64[us, UTC]")
    )

    # Reorder so entity + timestamp lead. This order matters — Feast's
    # offline_write_batch validates the pyarrow schema strictly.
    cols = ["invoice_id", "event_timestamp", *FEATURE_COLUMNS]
    feat = feat[cols].reset_index(drop=True)

    # Always write the local copy — Feast's FileSource reads from it. Use
    # pyarrow directly with an explicit schema to bypass pandas' habit of
    # smuggling in __index_level_0__ and to lock column order.
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("invoice_id", pa.string()),
        ("event_timestamp", pa.timestamp("us", tz="UTC")),
        *[(c, pa.float64()) for c in FEATURE_COLUMNS],
    ])
    table = pa.Table.from_pandas(feat, schema=schema, preserve_index=False)

    local_path = Path(local_path) if local_path is not None else LOCAL_PARQUET_PATH
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, local_path)

    if s3_client is not None and bucket:
        buf = BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        s3_client.put_object(Bucket=bucket, Key=FEAST_PARQUET_KEY, Body=buf.getvalue())

    return feat, meta


# ---------------------------------------------------------------------------
# `feast apply` and `feast materialize-incremental` from Python
# ---------------------------------------------------------------------------


def feast_cli(*args, cwd: Path = FEAST_REPO_PATH, check: bool = True) -> str:
    """Run `feast <args>` in the feast_repo directory and stream output.

    Calls the `feast` console-script binary (`/opt/app-root/bin/feast` in
    the RHOAI workbench image). `python -m feast` does NOT work — the SDK
    package has no `__main__`, only a setuptools entry point.
    """
    cmd = ["feast", *args]
    proc = subprocess.run(
        cmd, cwd=str(cwd), check=False, capture_output=True, text=True
    )
    # Always print captured output so failures are debuggable; raise after.
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc.stdout


def apply_feature_definitions() -> None:
    """Refresh the local registry from the .py files. Idempotent."""
    feast_cli("apply")


def materialize_incremental(end_ts: Optional[pd.Timestamp] = None) -> None:
    """Load features from the offline parquet into the online store.

    NB: We deliberately AVOID `feast materialize-incremental` here.
    Feast 0.62's LocalComputeEngine writes intermediate results back to the
    offline store with a re-ordered pyarrow schema, which trips its own
    schema validator and fails with
    `ValueError: The input pyarrow table has schema ... with the incorrect
    columns`. Switching back to the SDK-level batch push (read the source
    parquet ourselves, hand the DataFrame to `write_to_online_store`)
    bypasses the buggy compute engine and works on the same offline data
    the FileSource points at.

    `end_ts` is accepted for API compatibility with the CLI version but is
    ignored — pushing the whole local parquet is fine at workshop scale
    (~thousands of rows). For production-scale incremental writes, this
    helper would track its own watermark instead.
    """
    del end_ts  # unused — see docstring
    fs = open_local_store()
    fv_name = "invoice_features"
    fv = fs.get_feature_view(fv_name)
    # Resolve the offline parquet path relative to the feast_repo
    # (Feast's FileSource stores its `path` as given in data_sources.py).
    source_path = Path(fv.batch_source.path)
    if not source_path.is_absolute():
        source_path = FEAST_REPO_PATH / source_path
    print(f"  reading offline parquet: {source_path}")
    df = pd.read_parquet(source_path)
    print(f"  push {len(df):,} rows to online store …")
    fs.write_to_online_store(fv_name, df)
    print("  done.")


# ---------------------------------------------------------------------------
# Convenience: open a FeatureStore handle for the local feast_repo
# ---------------------------------------------------------------------------


def open_local_store():
    """Return a Feast FeatureStore pinned at this repo's feast_repo dir."""
    from feast import FeatureStore  # lazy import — only present in workbench
    return FeatureStore(repo_path=str(FEAST_REPO_PATH))
