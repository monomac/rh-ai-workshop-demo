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

from kfp import dsl, compiler, kubernetes
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Name of the data-connection secret that `deploy/03-data-connection.yaml`
# provisions. It carries AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_S3_ENDPOINT / AWS_S3_BUCKET / AWS_DEFAULT_REGION — the env vars boto3
# auto-reads.
S3_SECRET = "aws-connection-rhoai-invoices"
S3_SECRET_KEYS = {
    "AWS_ACCESS_KEY_ID":     "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT":       "AWS_S3_ENDPOINT",
    "AWS_S3_BUCKET":         "AWS_S3_BUCKET",
    "AWS_DEFAULT_REGION":    "AWS_DEFAULT_REGION",
}


# RHOAI 3.4 GA pipeline runtime — Python 3.12 / RHEL 9. Digest matches
# RELATED_IMAGE_ODH_PIPELINE_RUNTIME_DATASCIENCE_CPU_PY312_IMAGE on
# rhods-operator.3.4.0 (verified live against sa-ai 2026-05-16).
BASE_IMAGE = "registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:ed6634540d78910ceedc826b871641fb3f66b27be45b50df31c504582204a661"

# Default external route of `elos-model-registry` in `rhoai-model-registries`.
# In-cluster Service is NetworkPolicy-restricted to the router, so pipeline pods
# must talk to the public route — same as the workbench's MODEL_REGISTRY_URL.
DEFAULT_REGISTRY_URL = "https://elos-model-registry-rest.apps.rosa.sa-ai.oidv.p3.openshiftapps.com"


# ---------------------------------------------------------------------------
# Component: ingest
# ---------------------------------------------------------------------------

# The base runtime image already ships pandas / numpy / scikit-learn / joblib,
# so we only `packages_to_install` what's missing (boto3, requests). Pins are
# floors instead of exact-equal because RHOAI's PyPI mirror only carries
# Red Hat-blessed versions (e.g. boto3 1.35+, not 1.34).
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3>=1.35,<2.0"],
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
    # pandas / numpy / scikit-learn / joblib all ship in the base runtime image.
    packages_to_install=[],
)
def train_model(
    in_dataset: Input[Dataset],
    contamination: float,
    n_estimators: int,
    random_state: int,
    out_model: Output[Model],
    out_metrics: Output[Metrics],
) -> float:
    """Train Isolation Forest; also returns ROC-AUC so the gate can read it.

    We `log_metric` to `out_metrics` for the Dashboard's Metrics tab, AND
    return roc_auc as a value-typed output so the next task can branch on it
    without re-reading the artifact (Input[Metrics] artifacts in KFP v2 live
    in MLMD, not as a file at `.path` — so reading them by file is fragile).
    """
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
    roc_auc = 0.0
    if y is not None and y.sum() > 0:
        roc_auc = float(roc_auc_score(y, scores))
        metrics["roc_auc"] = roc_auc
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
    return roc_auc


# ---------------------------------------------------------------------------
# Component: evaluate (gate before promotion)
# ---------------------------------------------------------------------------

@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(roc_auc: float, min_roc_auc: float) -> bool:
    """Gate: pass only if ROC-AUC meets the configured threshold."""
    decision = roc_auc >= min_roc_auc
    print(f"roc_auc={roc_auc:.3f}, threshold={min_roc_auc:.3f} -> promote={decision}")
    return decision


# ---------------------------------------------------------------------------
# Component: register (publish to S3 + Model Registry)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["boto3>=1.35,<2.0", "requests>=2.32,<3.0"],
)
def register_model(
    in_model: Input[Model],
    s3_bucket: str,
    model_name: str,
    registry_url: str,
) -> str:
    """Copy the model to a versioned S3 key and register it in Model Registry.

    Auth pattern mirrors notebook 03 §3.5 — bearer SA token + TLS via the
    cluster's custom CA bundle. The pipeline pod's SA is bound to
    `registry-user-elos-model-registry` via `deploy/07-model-registry-rbac.yaml`.
    `?name=` is ignored by v1alpha3, so we list+filter client-side, then create
    idempotently. Version POST requires `registeredModelId` in the body.
    """
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
    base_key = f"models/{model_name}/{version}"

    # train_model writes a *dict* bundle (pipeline + metadata) to in_model.path,
    # which the notebooks know how to unpack. MLServer's sklearn runtime,
    # however, expects the joblib to deserialize directly to an estimator
    # with a `.predict()` method. So we split:
    #   - model.joblib   — just the sklearn Pipeline (deploy-ready)
    #   - bundle.joblib  — the original dict (for notebooks and audit)
    import joblib  # noqa: E402
    bundle = joblib.load(in_model.path)
    sk_pipeline = bundle["pipeline"] if isinstance(bundle, dict) else bundle

    import io
    sklearn_buf = io.BytesIO()
    joblib.dump(sk_pipeline, sklearn_buf)
    s3.put_object(Bucket=s3_bucket, Key=f"{base_key}/model.joblib",
                  Body=sklearn_buf.getvalue())
    with open(in_model.path, "rb") as f:
        s3.put_object(Bucket=s3_bucket, Key=f"{base_key}/bundle.joblib",
                      Body=f.read())
    s3_uri = f"s3://{s3_bucket}/{base_key}/"
    print("uploaded:", s3_uri)
    print("  - model.joblib  (sklearn Pipeline, MLServer-loadable)")
    print("  - bundle.joblib (full dict bundle for notebooks)")

    registry_url = (registry_url or "").rstrip("/")
    if not registry_url:
        print("registry_url empty — skipping Model Registry call.")
        return s3_uri

    TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    CA_BUNDLE = "/etc/pki/tls/custom-certs/ca-bundle.crt"
    API = f"{registry_url}/api/model_registry/v1alpha3"

    if not os.path.exists(TOKEN_PATH):
        print(f"SA token not at {TOKEN_PATH} — registry call skipped.")
        return s3_uri
    with open(TOKEN_PATH) as f:
        token = f.read().strip()
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    verify = CA_BUNDLE if os.path.exists(CA_BUNDLE) else True

    try:
        # 1) Ensure the registered model exists (list + client-side filter,
        #    because v1alpha3 ignores ?name=).
        page = requests.get(f"{API}/registered_models",
                            headers=headers, verify=verify, timeout=10)
        page.raise_for_status()
        rm_id = None
        for m in page.json().get("items", []):
            if m.get("name") == model_name:
                rm_id = m["id"]
                break
        if rm_id is None:
            r = requests.post(
                f"{API}/registered_models",
                headers=headers, verify=verify, timeout=10,
                json={
                    "name": model_name,
                    "description": "Isolation Forest pro detekci anomálií "
                                   "ve fakturách (auto-promoted by pipeline).",
                    "owner": "controlling@example.com",
                },
            )
            r.raise_for_status()
            rm_id = r.json()["id"]
            print(f"registered_model created: {model_name} (id={rm_id})")
        else:
            print(f"registered_model exists: {model_name} (id={rm_id})")

        # 2) Create a model version tied to this pipeline run.
        mv = {
            "name": f"pipeline-{version}",
            "description": f"Pipeline build {version} ({s3_uri})",
            "state": "LIVE",
            "author": "ds-pipelines",
            "registeredModelId": rm_id,  # required in body too, not just path
            "customProperties": {
                "s3_uri": {
                    "string_value": s3_uri,
                    "metadataType": "MetadataStringValue",
                },
                "source": {
                    "string_value": "data-science-pipelines",
                    "metadataType": "MetadataStringValue",
                },
            },
        }
        v = requests.post(f"{API}/registered_models/{rm_id}/versions",
                          headers=headers, json=mv, verify=verify, timeout=10)
        if v.status_code == 201:
            print(f"model_version created: {mv['name']} (id={v.json()['id']})")
        elif v.status_code == 409:
            print(f"model_version exists: {mv['name']}")
        else:
            print(f"version POST -> {v.status_code}: {v.text[:200]}")
    except Exception as e:  # noqa: BLE001
        # Non-fatal: S3 publish already succeeded; surface the error for the
        # Dashboard logs but don't fail the pipeline run.
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
    registry_url: str = DEFAULT_REGISTRY_URL,
):
    ingest = ingest_invoices(s3_bucket=s3_bucket, s3_key=s3_key)
    # Inject AWS creds from the data-connection secret so boto3 can talk to MinIO.
    kubernetes.use_secret_as_env(
        ingest, secret_name=S3_SECRET, secret_key_to_env=S3_SECRET_KEYS,
    )

    train = train_model(
        in_dataset=ingest.outputs["out_dataset"],
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    evaluate = evaluate_model(
        roc_auc=train.outputs["Output"],
        min_roc_auc=min_roc_auc,
    )

    with dsl.If(evaluate.output == True, name="promote-if-passing"):
        register = register_model(
            in_model=train.outputs["out_model"],
            s3_bucket=s3_bucket,
            model_name=model_name,
            registry_url=registry_url,
        )
        # register_model writes the model joblib back to S3 and posts to the
        # Model Registry, so it needs the same AWS creds.
        kubernetes.use_secret_as_env(
            register, secret_name=S3_SECRET, secret_key_to_env=S3_SECRET_KEYS,
        )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=invoice_anomaly_pipeline,
        package_path="invoice_anomaly_pipeline.yaml",
    )
    print("compiled -> invoice_anomaly_pipeline.yaml")
