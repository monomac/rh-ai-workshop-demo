"""Shared feature engineering and IO helpers for the invoice anomaly demo.

Keeping this in a small module means the workshop notebooks stay readable
and the same features are used at train and score time (reproducibility).
"""

from __future__ import annotations

import os
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_invoices(path_or_key: str, s3_client=None, bucket: Optional[str] = None) -> pd.DataFrame:
    """Load an invoices CSV either from local disk or S3/MinIO.

    If `s3_client` is provided, `path_or_key` is treated as an object key in
    `bucket`. Otherwise it is read directly from local disk.
    """
    if s3_client is not None and bucket:
        obj = s3_client.get_object(Bucket=bucket, Key=path_or_key)
        df = pd.read_csv(BytesIO(obj["Body"].read()))
    else:
        df = pd.read_csv(path_or_key)
    df["issued_on"] = pd.to_datetime(df["issued_on"]).dt.date
    df["due_on"] = pd.to_datetime(df["due_on"]).dt.date
    return df


def write_csv_s3(df: pd.DataFrame, s3_client, bucket: str, key: str) -> None:
    """Write a dataframe as CSV to S3/MinIO."""
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# Czech public holidays we account for. Keep in sync with generator.
HOLIDAYS_2025 = {
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 21),
    date(2025, 5, 1), date(2025, 5, 8), date(2025, 7, 5),
    date(2025, 7, 6), date(2025, 9, 28), date(2025, 10, 28),
    date(2025, 11, 17), date(2025, 12, 24), date(2025, 12, 25),
    date(2025, 12, 26),
}

FEATURE_COLUMNS = [
    "log_amount",
    "amount_zscore_in_category",
    "is_weekend",
    "is_holiday",
    "missing_po",
    "vendor_frequency",
    "round_sum_flag",
    "days_to_due",
    "category_idx",
]


def _is_weekend(d) -> int:
    return int(pd.Timestamp(d).weekday() >= 5)


def _is_holiday(d) -> int:
    return int(d in HOLIDAYS_2025)


def _round_sum(amount: float) -> int:
    """Crude 'round number' detector: divisible by 10 000 with no decimals."""
    return int(round(amount, 2) % 10_000 == 0 and amount >= 10_000)


def build_features(
    df: pd.DataFrame,
    category_index: Optional[dict] = None,
    vendor_freq_table: Optional[pd.Series] = None,
    category_stats: Optional[pd.DataFrame] = None,
):
    """Return (X, meta) where X is the feature matrix and meta is the lookup
    tables produced from this dataframe. At train time the caller passes
    `None` for the lookup tables and they are built from data. At score time
    the caller passes the tables produced at train time so features are
    encoded identically.
    """
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount_czk"])

    # vendor frequency — how often have we seen this vendor before?
    if vendor_freq_table is None:
        vendor_freq_table = df["vendor"].value_counts()
    df["vendor_frequency"] = df["vendor"].map(vendor_freq_table).fillna(0).astype(int)

    # category amount stats — z-score within category
    if category_stats is None:
        category_stats = df.groupby("category")["amount_czk"].agg(["mean", "std"]).fillna(1)
    df = df.merge(category_stats.rename(columns={"mean": "_cat_mean", "std": "_cat_std"}),
                  left_on="category", right_index=True, how="left")
    df["_cat_std"] = df["_cat_std"].replace(0, 1).fillna(1)
    df["amount_zscore_in_category"] = (df["amount_czk"] - df["_cat_mean"]) / df["_cat_std"]

    # boolean / categorical features
    df["is_weekend"] = df["issued_on"].apply(_is_weekend)
    df["is_holiday"] = df["issued_on"].apply(_is_holiday)
    df["missing_po"] = df["po_number"].fillna("").astype(str).eq("").astype(int)
    df["round_sum_flag"] = df["amount_czk"].apply(_round_sum)
    df["days_to_due"] = (pd.to_datetime(df["due_on"]) - pd.to_datetime(df["issued_on"])).dt.days

    if category_index is None:
        cats = sorted(df["category"].unique())
        category_index = {c: i for i, c in enumerate(cats)}
    df["category_idx"] = df["category"].map(category_index).fillna(-1).astype(int)

    X = df[FEATURE_COLUMNS].astype(float).values
    meta = {
        "category_index": category_index,
        "vendor_freq_table": vendor_freq_table,
        "category_stats": category_stats,
    }
    return X, meta


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

def explain_row(row: pd.Series) -> str:
    """Human-readable reason codes a controller can act on.

    Used at score time to enrich the model's anomaly_score with the *why*
    so the business user does not have to interpret a raw model output.
    """
    reasons = []
    if row.get("missing_po", 0) and row.get("amount_czk", 0) > 100_000:
        reasons.append("vysoká částka bez PO")
    if row.get("round_sum_flag", 0):
        reasons.append("podezřele kulatá částka")
    if row.get("is_weekend", 0) or row.get("is_holiday", 0):
        reasons.append("zaúčtováno o víkendu / svátku")
    if abs(row.get("amount_zscore_in_category", 0)) > 3:
        reasons.append("částka mimo obvyklý rozsah kategorie")
    if row.get("vendor_frequency", 1) <= 1 and row.get("amount_czk", 0) > 100_000:
        reasons.append("nový dodavatel s vysokou částkou")
    return "; ".join(reasons) if reasons else "kombinace signálů (viz feature attribution)"


# ---------------------------------------------------------------------------
# Convenience: get S3/MinIO client from RHOAI Data Connection env vars
# ---------------------------------------------------------------------------

def get_s3_client():
    """Build a boto3 client from the env vars RHOAI 'Data Connection' injects.

    The standard RHOAI Data Connection secret produces these variables on the
    workbench: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT,
    AWS_S3_BUCKET, AWS_DEFAULT_REGION.
    """
    import boto3  # imported lazily so the module loads without boto3 locally

    endpoint = os.environ.get("AWS_S3_ENDPOINT")
    region   = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client(
        "s3",
        endpoint_url=endpoint if endpoint else None,
        region_name=region,
    )


def get_bucket() -> str:
    return os.environ["AWS_S3_BUCKET"]
