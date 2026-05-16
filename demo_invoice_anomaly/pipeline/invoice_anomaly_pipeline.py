"""Data Science Pipeline for the invoice anomaly demo.

Authored against Kubeflow Pipelines SDK v2 (kfp >= 2.7), which is what
Red Hat AI / RHOAI 3.4 Data Science Pipelines ships under the hood.

Compile with:
    python invoice_anomaly_pipeline.py
    # produces invoice_anomaly_pipeline.yaml

Upload via the RHOAI UI (Data Science Pipelines → Import pipeline)
or via the kfp CLI:
    kfp pipeline upload --pipeline-name 'invoice-anomaly' \\
        invoice_anomaly_pipeline.yaml

The pipeline orchestrates the same logic that's in the workshop notebooks,
but as a scheduled job: ingest -> train -> evaluate -> register.
"""

from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics


BASE_IMAGE = "image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/s2i-generic-data-science-notebook:2024.2"


# ---------------------------------------------------------------------------
# Component: ingest
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.*", "boto3==1.34.*"],
)
def ingest_invoices(
    s3_bucket: str,
    s3_key: str,
    out_dataset: Output[Dataset],
) -> None:
    """Pull the latest invoices CSV from MinIO/S3 and stage it for training."""
    import os
    import boto3
    import pandas as pd

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_S3_ENDPOINT"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    body = s3.get_object(Bucket=s3_bucket, Key=s3_key)["Body"].read()
    out_path = out_dataset.path + ".csv"
    with open(out_path, "wb") as f:
        f.write(body)
    out_dataset.path = out_path
    df = pd.read_csv(out_path)
    print(f"ingested {len(df)} rows from s3://{s3_bucket}/{s3_key}")


# ---------------------------------------------------------------------------
# Component: train
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas==2.2.*", "numpy==1.26.*", "scikit-learn==1.4.*",
        "joblib==1.4.*",
    ],
)
def train_model(
    in_dataset: Input[Dataset],
    contamination: float,
    n_estimators: int,
    random_state: int,
    out_model: Output[Model],
    out_metrics: Output[Metrics],
) -> None:
    """Train Isolation Forest on the staged invoices dataset."""
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from datetime import date
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(in_dataset.path)
    df["issued_on"] = pd.to_datetime(df["issued_on"]).dt.date
    df["due_on"]    = pd.to_datetime(df["due_on"]).dt.date

    # Inline feature engineering — kept short, mirrors src/invoice_features.py
    df["log_amount"]    = np.log1p(df["amount_czk"])
    cat_stats = df.groupby("category")["amount_czk"].agg(["mean", "std"]).fillna(1)
    cat_stats["std"]    = cat_stats["std"].replace(0, 1)
    df = df.merge(cat_stats.rename(columns={"mean": "_m", "std": "_s"}),
                  left_on="category", right_index=True, how="left")
    df["amount_zscore_in_category"] = (df["amount_czk"] - df["_m"]) / df["_s"]
    df["is_weekend"]     = pd.to_datetime(df["issued_on"]).dt.weekday.ge(5).astype(int)
    df["missing_po"]     = df["po_number"].fillna("").astype(str).eq("").astype(int)
    df["round_sum_flag"] = ((df["amount_czk"] % 10_000 == 0) & (df["amount_czk"] >= 10_000)).astype(int)
    df["days_to_due"]    = (pd.to_datetime(df["due_on"]) - pd.to_datetime(df["issued_on"])).dt.days
    vfreq = df["vendor"].value_counts()
    df["vendor_frequency"] = df["vendor"].map(vfreq).fillna(0)
    cat_idx = {c: i for i, c in enumerate(sorted(df["category"].unique()))}
    df["category_idx"]   = df["category"].map(cat_idx)

    feature_cols = ["log_amount", "amount_zscore_in_category", "is_weekend",
                    "missing_po", "vendor_frequency", "round_sum_flag",
                    "days_to_due", "category_idx"]
    X = df[feature_cols].astype(float).values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(contamination=contamination,
                                    n_estimators=n_estimators,
                                    random_state=random_state)),
    ])
    pipe.fit(X)

    scores = -pipe.decision_function(X)
    y      = df["is_anomaly"].values if "is_anomaly" in df.columns else None

    metrics = {"n_rows": len(df)}
    if y is not None and y.sum() > 0:
        metrics["roc_auc"] = float(roc_auc_score(y, scores))
        metrics["pr_auc"]  = float(average_precision_score(y, scores))

    joblib.dump({
        "pipeline": pipe,
        "feature_columns": feature_cols,
        "category_index": cat_idx,
        "vendor_freq_table": vfreq.to_dict(),
        "category_stats": cat_stats.to_dict(orient="index"),
        "metrics": metrics,
        "trained_on": str(date.today()),
    }, out_model.path)

    for k, v in metrics.items():
        out_metrics.log_metric(k, v)
    print("metrics:", json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Component: evaluate (gate before promotion)
# ---------------------------------------------------------------------------

@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(
    in_metrics: Input[Metrics],
    min_roc_auc: float,
) -> bool:
    """Gate: pass only if ROC-AUC meets the configured threshold."""
    import json
    raw = open(in_metrics.path).read()
    print("metrics file:", raw)
    data = json.loads(raw)
    # KFP metrics file shape: {"metrics": [{"name": "roc_auc", "numberValue": ...}, ...]}
    by_name = {m["name"]: m.get("numberValue", 0) for m in data.get("metrics", [])}
    roc = by_name.get("roc_auc", 0)
    decision = roc >= min_roc_auc
    print(f"roc_auc={roc:.3f}, threshold={min_roc_auc:.3f} -> promote={decision}")
    return decision


# ---------------------------------------------------------------------------
# Component: register (publish to S3 + Model Registry)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3==1.34.*", "requests==2.32.*"],
)
def register_model(
    in_model: Input[Model],
    s3_bucket: str,
    model_name: str,
    registry_url: str,
) -> str:
    """Copy the model to a versioned S3 key and register it in Model Registry."""
    import os
    import datetime as dt
    import boto3
    import requests

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_S3_ENDPOINT"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    version = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    key = f"models/{model_name}/{version}/model.joblib"
    with open(in_model.path, "rb") as f:
        s3.put_object(Bucket=s3_bucket, Key=key, Body=f.read())
    s3_uri = f"s3://{s3_bucket}/{key}"
    print("uploaded:", s3_uri)

    if registry_url:
        try:
            payload = {
                "name": model_name,
                "description": f"Invoice anomaly model {version} (auto-promoted by pipeline)",
                "owner": "controlling@example.com",
            }
            r = requests.post(
                f"{registry_url}/api/model_registry/v1alpha3/registered_models",
                json=payload, timeout=10,
            )
            print("registry response:", r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            print("registry call failed (non-fatal):", e)
    return s3_uri


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dsl.pipeline(
    name="invoice-anomaly",
    description="Trénink a registrace modelu pro detekci anomálií ve fakturách.",
)
def invoice_anomaly_pipeline(
    s3_bucket: str = "rhoai-workshop-invoices",
    s3_key: str = "invoices.csv",
    contamination: float = 0.03,
    n_estimators: int = 200,
    random_state: int = 42,
    min_roc_auc: float = 0.85,
    model_name: str = "invoice-anomaly-detector",
    registry_url: str = "http://model-registry-service:8080",
):
    ingest = ingest_invoices(s3_bucket=s3_bucket, s3_key=s3_key)

    train = train_model(
        in_dataset=ingest.outputs["out_dataset"],
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    evaluate = evaluate_model(
        in_metrics=train.outputs["out_metrics"],
        min_roc_auc=min_roc_auc,
    )

    with dsl.If(evaluate.output == True, name="promote-if-passing"):
        register_model(
            in_model=train.outputs["out_model"],
            s3_bucket=s3_bucket,
            model_name=model_name,
            registry_url=registry_url,
        )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=invoice_anomaly_pipeline,
        package_path="invoice_anomaly_pipeline.yaml",
    )
    print("compiled -> invoice_anomaly_pipeline.yaml")
