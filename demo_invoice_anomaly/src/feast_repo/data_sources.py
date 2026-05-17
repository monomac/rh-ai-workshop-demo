"""Feast data sources for the invoice-anomaly demo.

Single offline source: a parquet of *already-computed* feature values living
in MinIO. The training notebook (02) writes this parquet once, the Feast SDK
reads it for both historical joins (training) and materialization into the
online store (serving).

Why pre-computed and not OnDemandFeatureView?
- IsolationForest needs a tiny number of fitted lookup tables (vendor
  frequency, category z-score statistics). Those live in the trained
  `model.joblib`, not in raw invoices. Pushing them through a UDF inside an
  ODFV would force a circular dependency between the model and the FV.
- Pre-computing in the training notebook keeps `invoice_features.py` as the
  single source of truth for transformation logic, while Feast owns the
  registry, point-in-time guarantees, and online serving.
"""

import os

from feast import FileSource
from feast.data_format import ParquetFormat

# The path to the feature parquet. Two environments read it:
#   - Operator-side `feast apply` only needs the path string to register it;
#     it does NOT read data at apply time.
#   - Workbench-side `feast get-historical-features` reads the parquet over
#     S3/MinIO. The AWS_S3_ENDPOINT env (from the Data Connection) is passed
#     through `s3_endpoint_override` so the MinIO HTTP endpoint is used
#     instead of `https://s3.amazonaws.com`.
_BUCKET = os.environ.get("AWS_S3_BUCKET", "rhoai-workshop-invoices")
_FEAST_PARQUET_KEY = "feast/invoice_features.parquet"

invoice_features_source = FileSource(
    name="invoice_features_source",
    path=f"s3://{_BUCKET}/{_FEAST_PARQUET_KEY}",
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    s3_endpoint_override=os.environ.get("AWS_S3_ENDPOINT") or None,
    description=(
        "Per-invoice feature vectors written by notebook 02 / the training "
        "pipeline. One row per invoice_id, event_timestamp = issued_on."
    ),
)
