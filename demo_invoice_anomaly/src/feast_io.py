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

# S3 key the FileSource is configured to read. Stays in sync with
# data_sources.py — change here AND there.
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
    local_path  : if given, the parquet is also written to disk at this path
                  (useful for in-cluster pipelines where local IO is faster).

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

    feat = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    feat["invoice_id"] = invoices_df["invoice_id"].values
    feat["event_timestamp"] = pd.to_datetime(invoices_df["issued_on"]).dt.tz_localize("UTC")

    # Reorder so entity + timestamp lead.
    cols = ["invoice_id", "event_timestamp", *FEATURE_COLUMNS]
    feat = feat[cols]

    if local_path is not None:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(local_path, index=False)

    if s3_client is not None and bucket:
        buf = BytesIO()
        feat.to_parquet(buf, index=False)
        buf.seek(0)
        s3_client.put_object(Bucket=bucket, Key=FEAST_PARQUET_KEY, Body=buf.getvalue())

    return feat, meta


# ---------------------------------------------------------------------------
# `feast apply` and `feast materialize-incremental` from Python
# ---------------------------------------------------------------------------


def feast_cli(*args, cwd: Path = FEAST_REPO_PATH, check: bool = True) -> str:
    """Run `feast <args>` in the feast_repo directory and stream output."""
    cmd = [sys.executable, "-m", "feast", *args]
    proc = subprocess.run(
        cmd, cwd=str(cwd), check=check, capture_output=True, text=True
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.stdout


def apply_feature_definitions() -> None:
    """Refresh the local registry from the .py files. Idempotent."""
    feast_cli("apply")


def materialize_incremental(end_ts: Optional[pd.Timestamp] = None) -> None:
    """Load features from the offline parquet into the online store, up to
    `end_ts` (defaults to now). Incremental — Feast remembers the last
    materialize watermark per FV."""
    end = (end_ts or pd.Timestamp.utcnow()).isoformat()
    feast_cli("materialize-incremental", end)


# ---------------------------------------------------------------------------
# Convenience: open a FeatureStore handle for the local feast_repo
# ---------------------------------------------------------------------------


def open_local_store():
    """Return a Feast FeatureStore pinned at this repo's feast_repo dir."""
    from feast import FeatureStore  # lazy import — only present in workbench
    return FeatureStore(repo_path=str(FEAST_REPO_PATH))
