"""Feast data sources for the invoice-anomaly demo.

Single offline source: a parquet of *already-computed* feature values
sitting next to the feast project (`feast_repo/data/invoice_features.parquet`).
The training notebook (02) writes the parquet, the Feast SDK reads it for
both historical joins (training) and materialization into the online store
(serving).

Why local-path instead of `s3://...`?
- Feast's Dask offline engine reads parquet via fsspec, which needs `s3fs`
  for s3:// URIs. The RHOAI workbench image doesn't ship s3fs, and adding
  a runtime pip-install in a workshop notebook is a portability tripwire.
- The operator-side `feast apply` doesn't read data, just records the
  schema, so the path string doesn't have to be operator-reachable.
- Notebook 02 writes the parquet to S3/MinIO *and* to the local feast_repo
  path: S3 for cross-environment audit, local for Feast offline reads.

Why pre-computed and not OnDemandFeatureView?
- IsolationForest needs a tiny number of fitted lookup tables (vendor
  frequency, category z-score statistics). Those live in the trained
  `model.joblib`, not in raw invoices. Pushing them through a UDF inside an
  ODFV would force a circular dependency between the model and the FV.
- Pre-computing in the training notebook keeps `invoice_features.py` as the
  single source of truth for transformation logic, while Feast owns the
  registry, point-in-time guarantees, and online serving.
"""

from feast import FileSource
from feast.data_format import ParquetFormat

# Resolved relative to wherever `feast apply` is invoked — for the workbench
# that's src/feast_repo/, so this becomes src/feast_repo/data/invoice_features.parquet
# (the same `data/` dir as the sqlite registry + online store). Operator does
# its own thing with /feast-data/ and never reads this path at apply time.
LOCAL_PARQUET_PATH = "data/invoice_features.parquet"

invoice_features_source = FileSource(
    name="invoice_features_source",
    path=LOCAL_PARQUET_PATH,
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    description=(
        "Per-invoice feature vectors written by notebook 02 / the training "
        "pipeline. One row per invoice_id, event_timestamp = issued_on. "
        "Mirrored to s3://$AWS_S3_BUCKET/feast/invoice_features.parquet for "
        "audit and cross-environment access; the FileSource here points at "
        "the local copy because Feast's Dask offline engine needs s3fs to "
        "talk to S3 and the workbench image doesn't ship it."
    ),
)
