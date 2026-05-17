"""Feast feature views for the invoice-anomaly demo.

A single FeatureView exposes the nine features from
`invoice_features.FEATURE_COLUMNS`. Keep these in lock-step with the schema
in ../invoice_features.py — if you add a feature there, register it here.
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, Int32

from entities import invoice
from data_sources import invoice_features_source

invoice_features_fv = FeatureView(
    name="invoice_features",
    entities=[invoice],
    # 1 year TTL — past this point Feast treats a feature value as stale and
    # will not include it in a point-in-time join. Invoices have natural
    # 30-60 day lifecycles so 365d is comfortably permissive for the demo.
    ttl=timedelta(days=365),
    source=invoice_features_source,
    online=True,
    offline=True,
    schema=[
        # log(amount_czk + 1) — keeps the long-tailed amount distribution
        # tractable for tree-based models like IsolationForest.
        Field(name="log_amount", dtype=Float32),
        # (amount - category_mean) / category_std — flags amounts that are
        # outliers _within_ their category (a 1 M Kč office supply order is
        # weirder than a 1 M Kč software licence).
        Field(name="amount_zscore_in_category", dtype=Float32),
        Field(name="is_weekend", dtype=Int32),
        Field(name="is_holiday", dtype=Int32),
        Field(name="missing_po", dtype=Int32),
        Field(name="vendor_frequency", dtype=Int32),
        Field(name="round_sum_flag", dtype=Int32),
        Field(name="days_to_due", dtype=Int32),
        Field(name="category_idx", dtype=Int32),
    ],
    description=(
        "Per-invoice features used by the IsolationForest anomaly detector. "
        "Computed from raw invoice rows by src/invoice_features.build_features "
        "and written to the offline parquet by the training notebook / pipeline."
    ),
    tags={
        "team": "controlling",
        "use_case": "invoice-anomaly-detection",
    },
)
