"""Build the three workshop notebooks as .ipynb files.

We assemble the notebooks programmatically so the source-of-truth is plain
Python here rather than handwritten JSON. The CZ narrative goes in markdown
cells, EN technical comments go in code cells.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def _cell_id(kind: str, lines) -> str:
    """Deterministic 8-char cell id derived from contents — required by
    nbformat 4.5+. Stable across rebuilds so diffs stay clean."""
    h = hashlib.sha1((kind + "\n".join(lines)).encode("utf-8")).hexdigest()
    return h[:8]


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id("md", lines),
        "metadata": {},
        "source": _src(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id("code", lines),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines):
    joined = "\n".join(lines)
    # Jupyter expects a list of strings with newlines preserved
    return [ln + ("\n" if i < len(joined.splitlines()) - 1 else "")
            for i, ln in enumerate(joined.splitlines())]


def write_notebook(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
# Notebook 01 — Exploration (controller POV)
# ---------------------------------------------------------------------------

nb01 = [
    md(
        "# 01 · Průzkum faktur — pohled controllera",
        "",
        "**Scénář ze slide 20:** _self-service pro analytiky_ · _rychlejší prototypování_",
        "",
        "Tento notebook simuluje situaci, kdy controller dostane úkol \"podívej se,",
        "jestli v posledních fakturách nejsou nějaké podezřelé položky\".",
        "Bez čekání na BI tým si otevře workbench na Red Hat AI, načte data z MinIO",
        "a během minut má první obrázek, kde hledat.",
        "",
        "Žádný kód do produkce — jen rychlý průzkum, který by jinak zabral týden.",
    ),
    md(
        "## 1.1 Připojení k datům",
        "",
        "Workbench používá _Data Connection_ — RHOAI nám pomocí proměnných prostředí",
        "předá přístup k MinIO bucketu, kde leží exportované faktury z Pohody.",
    ),
    code(
        "import os, sys",
        "sys.path.insert(0, '../src')  # umožní import sdílených helperů",
        "",
        "from invoice_features import load_invoices",
        "",
        "# When running on RHOAI workbench, Data Connection sets these env vars.",
        "# Locally we fall back to the bundled CSV so the notebook is portable.",
        "if os.environ.get('AWS_S3_BUCKET'):",
        "    from invoice_features import get_s3_client, get_bucket",
        "    s3 = get_s3_client()",
        "    df = load_invoices('invoices.csv', s3_client=s3, bucket=get_bucket())",
        "    print(f'Načteno z S3 bucketu {get_bucket()}')",
        "else:",
        "    df = load_invoices('../data/invoices.csv')",
        "    print('Načteno z lokálního CSV (mimo cluster)')",
        "",
        "print(f'Řádků: {len(df):,}')",
        "df.head()",
    ),
    md(
        "## 1.2 Co máme v datech",
        "",
        "Sloupce `is_anomaly` a `anomaly_type` jsou v reálu samozřejmě **neznámé** —",
        "tady je máme jen proto, abychom si na konci ověřili, že model najde to,",
        "co skutečně vadí. V produkci je tam neuvidí ani analytik, ani model.",
    ),
    code(
        "df.dtypes",
    ),
    code(
        "# Quick overview — kolik faktur za kategorii, kolik celkem proteklo",
        "summary = (df",
        "    .groupby('category')",
        "    .agg(faktur=('invoice_id', 'count'),",
        "         celkem_czk=('amount_czk', 'sum'),",
        "         prum_czk=('amount_czk', 'mean'))",
        "    .sort_values('celkem_czk', ascending=False))",
        "summary",
    ),
    md(
        "## 1.3 Distribuce částek",
        "",
        "Klasická rychlá kontrola: log-stupnice, ať vidíme i extrémní hodnoty.",
        "Pokud se v pravém ocasu objeví ostrá špička, je to obvykle první stopa.",
    ),
    code(
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "",
        "fig, ax = plt.subplots(figsize=(9, 4))",
        "ax.hist(np.log10(df['amount_czk']), bins=60, color='#cc0000', alpha=0.7)",
        "ax.set_xlabel('log10(amount CZK)')",
        "ax.set_ylabel('počet faktur')",
        "ax.set_title('Distribuce výše faktur (log10)')",
        "plt.tight_layout(); plt.show()",
    ),
    md(
        "## 1.4 První náhled na podezřelé řádky",
        "",
        "Než pustíme model, podívejme se očima controllera na nejjednodušší red flags:",
        "",
        "- faktura **bez PO** s vysokou částkou,",
        "- **kulatá** částka (50 000, 100 000, …),",
        "- doklad zaúčtovaný o **víkendu** nebo svátku.",
        "",
        "Tyhle pravidla bychom napsali i v Excelu. Pointa demo je, že model najde",
        "i ty kombinace, na které žádné jednoduché pravidlo nedosáhne.",
    ),
    code(
        "import pandas as pd",
        "",
        "df['issued_dow'] = pd.to_datetime(df['issued_on']).dt.day_name()",
        "",
        "rule_hits = df[",
        "    ((df['po_number'].fillna('') == '') & (df['amount_czk'] > 100_000))",
        "    | (df['amount_czk'] % 10_000 == 0)",
        "    | (df['issued_dow'].isin(['Saturday', 'Sunday']))",
        "]",
        "print(f'Pravidlové red-flag řádky: {len(rule_hits)} z {len(df)} ({len(rule_hits)/len(df):.1%})')",
        "rule_hits.head(10)",
    ),
    md(
        "## 1.5 Co tím controller získal",
        "",
        "Za pět buněk a několik minut má:",
        "",
        "- napojení na živá data (žádný ticket na IT),",
        "- první rozdělení podle dodavatele a kategorie,",
        "- shortlist na základě pravidel.",
        "",
        "V dalším notebooku si analytik natrénuje model, který tyhle signály",
        "**kombinuje** a najde i to, co jednoduchá pravidla minou.",
    ),
]

write_notebook("01_explore_invoices.ipynb", nb01)


# ---------------------------------------------------------------------------
# Notebook 02 — Train + MLflow
# ---------------------------------------------------------------------------

nb02 = [
    md(
        "# 02 · Trénink modelu pro detekci anomálií",
        "",
        "**Scénář ze slide 20:** _reprodukovatelnost a sdílení_ · _auditovatelnost a compliance_",
        "",
        "Z průzkumového notebooku víme, na co se dívat. Teď natrénujeme model,",
        "který kombinuje **všechny** signály najednou a každé faktuře přidělí",
        "skóre.",
        "",
        "Vše, co teď spustíme, se zaloguje do **MLflow** — datasety, parametry,",
        "metriky, artefakty. To je naše evidence pro audit (NIS2 / AI Act):",
        "kdo, kdy, na čem natrénoval, s jakým výsledkem.",
    ),
    md(
        "## 2.1 Načtení dat a feature engineering",
        "",
        "Featury držíme v `src/invoice_features.py`, aby se trénink i inference",
        "shodly. Žádné copy-paste mezi notebooky.",
    ),
    code(
        "import os, sys",
        "sys.path.insert(0, '../src')",
        "",
        "from invoice_features import load_invoices, build_features, FEATURE_COLUMNS",
        "",
        "if os.environ.get('AWS_S3_BUCKET'):",
        "    from invoice_features import get_s3_client, get_bucket",
        "    s3 = get_s3_client()",
        "    df = load_invoices('invoices.csv', s3_client=s3, bucket=get_bucket())",
        "else:",
        "    df = load_invoices('../data/invoices.csv')",
        "",
        "print(f'Trénovací data: {len(df):,} faktur')",
    ),
    code(
        "X, meta = build_features(df)",
        "print('Feature columns:', FEATURE_COLUMNS)",
        "print('Shape:', X.shape)",
    ),
    md(
        "## 2.2 Trénink Isolation Forest",
        "",
        "Pro tabulární anomálie je _Isolation Forest_ rozumný start —",
        "rychlý, nepotřebuje labely, dobře interpretovatelný.",
        "Parametry necháme **otevřené v buňce**, ať si je business analytik",
        "může upravit bez čtení dokumentace.",
    ),
    code(
        "# ---- PARAMETRY (uprav, pokud chceš jiný profil) ------------------",
        "CONTAMINATION = 0.03    # očekávaný podíl anomálií (0–0.1)",
        "N_ESTIMATORS  = 200     # počet stromů — víc = stabilnější, pomalejší",
        "MAX_SAMPLES   = 'auto'  # 'auto' = min(256, n_samples)",
        "RANDOM_STATE  = 42",
        "# ------------------------------------------------------------------",
    ),
    code(
        "from sklearn.ensemble import IsolationForest",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.pipeline import Pipeline",
        "",
        "pipe = Pipeline([",
        "    ('scaler', StandardScaler()),",
        "    ('iforest', IsolationForest(",
        "        contamination=CONTAMINATION,",
        "        n_estimators=N_ESTIMATORS,",
        "        max_samples=MAX_SAMPLES,",
        "        random_state=RANDOM_STATE,",
        "    )),",
        "])",
        "pipe.fit(X)",
        "print('Trénink hotov.')",
    ),
    md(
        "## 2.3 Vyhodnocení proti známým anomáliím",
        "",
        "V demo datech labely _máme_ — využijeme je k report-cardu. V produkci",
        "by se sem dosadila zpětná vazba schvalovatelů (controller reviewer flag).",
    ),
    code(
        "import numpy as np",
        "from sklearn.metrics import (precision_recall_curve, roc_auc_score,",
        "                              average_precision_score)",
        "",
        "scores = -pipe.decision_function(X)   # vyšší = podezřelejší",
        "y_true = df['is_anomaly'].values",
        "",
        "roc_auc = roc_auc_score(y_true, scores)",
        "pr_auc  = average_precision_score(y_true, scores)",
        "print(f'ROC-AUC: {roc_auc:.3f}')",
        "print(f'PR-AUC : {pr_auc:.3f}')",
    ),
    code(
        "# Recall@k — kolik anomálií zachytíme v top-k podezřelých?",
        "order = np.argsort(scores)[::-1]",
        "for k in (50, 100, 200, 500):",
        "    top_k = y_true[order[:k]]",
        "    print(f'Top {k:>4}: nalezeno {top_k.sum():>3} anomálií z {y_true.sum()} '",
        "          f'(precision {top_k.mean():.2%})')",
    ),
    md(
        "## 2.4 Logování do MLflow",
        "",
        "Tohle je krok, který dělá z notebooku **auditovatelný artefakt**.",
        "MLflow běží jako serving runtime v RHOAI — UI je dostupné na route",
        "`mlflow-server-<namespace>.apps.<cluster>`.",
    ),
    code(
        "try:",
        "    import mlflow",
        "    import mlflow.sklearn",
        "",
        "    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'file:./mlruns')",
        "    mlflow.set_tracking_uri(tracking_uri)",
        "    mlflow.set_experiment('invoice-anomaly')",
        "",
        "    with mlflow.start_run(run_name='isolation-forest-baseline') as run:",
        "        mlflow.log_params({",
        "            'contamination': CONTAMINATION,",
        "            'n_estimators': N_ESTIMATORS,",
        "            'max_samples': str(MAX_SAMPLES),",
        "            'features': ','.join(FEATURE_COLUMNS),",
        "            'n_train_rows': len(df),",
        "        })",
        "        mlflow.log_metrics({",
        "            'roc_auc': roc_auc,",
        "            'pr_auc': pr_auc,",
        "            'n_anomalies_true': int(y_true.sum()),",
        "        })",
        "        mlflow.sklearn.log_model(pipe, artifact_path='model')",
        "        run_id = run.info.run_id",
        "        print(f'MLflow run logged: {run_id}')",
        "        print(f'Tracking URI:     {tracking_uri}')",
        "except Exception as exc:",
        "    print('MLflow nedostupný v této session — pokračujeme bez něj:', exc)",
        "    run_id = None",
    ),
    md(
        "## 2.5 Uložení modelu (a meta tabulek) do S3",
        "",
        "Vedle MLflow registru ukládáme i přímo do bucketu — pro pipeline,",
        "která spouští skórování v nightly batchi, je to jednodušší než tahat",
        "binárku z MLflow.",
    ),
    code(
        "import joblib, json, io",
        "",
        "artifact = {",
        "    'pipeline': pipe,",
        "    'feature_columns': FEATURE_COLUMNS,",
        "    'category_index': meta['category_index'],",
        "    'vendor_freq_table': meta['vendor_freq_table'].to_dict(),",
        "    'category_stats': meta['category_stats'].to_dict(orient='index'),",
        "    'metrics': {'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)},",
        "    'mlflow_run_id': run_id,",
        "}",
        "",
        "out_path = '../data/invoice_anomaly_model.joblib'",
        "joblib.dump(artifact, out_path)",
        "print('Model uložen lokálně do:', out_path)",
        "",
        "if os.environ.get('AWS_S3_BUCKET'):",
        "    s3 = get_s3_client()",
        "    buf = io.BytesIO(); joblib.dump(artifact, buf); buf.seek(0)",
        "    s3.put_object(Bucket=get_bucket(), Key='models/invoice_anomaly_model.joblib',",
        "                  Body=buf.getvalue())",
        "    print('Model uložen i do S3 bucketu')",
    ),
    md(
        "## 2.6 Co máme z pohledu compliance",
        "",
        "- Notebook (kód) → Git.",
        "- Parametry + metriky + dataset signature → MLflow.",
        "- Binárka modelu → S3 (versionovaný klíč).",
        "- V dalším kroku model zaregistrujeme do **Model Registry** se stage",
        "  `Staging` → po review business sponzora `Production`.",
        "",
        "**To je celá audit trail jedním notebookem.**",
    ),
]

write_notebook("02_train_model.ipynb", nb02)


# ---------------------------------------------------------------------------
# Notebook 03 — Score + review queue + register
# ---------------------------------------------------------------------------

nb03 = [
    md(
        "# 03 · Skórování nových faktur a review queue",
        "",
        "**Scénář ze slide 20:** _most mezi daty a byznysem_ · _AI quickstart pattern_",
        "",
        "Tento notebook už cílí přímo na business uživatele — controller dostane",
        "**seřazený seznam faktur ke schválení / odmítnutí** s vysvětlením, proč",
        "je model označil. Žádné F1 score, žádné konfuzní matice.",
    ),
    md(
        "## 3.1 Načtení modelu a nových faktur",
    ),
    code(
        "import os, sys, joblib, io",
        "sys.path.insert(0, '../src')",
        "from invoice_features import (load_invoices, build_features, explain_row,",
        "                              FEATURE_COLUMNS)",
        "import pandas as pd",
        "",
        "# --- Načti model ---------------------------------------------------",
        "if os.environ.get('AWS_S3_BUCKET'):",
        "    from invoice_features import get_s3_client, get_bucket",
        "    s3 = get_s3_client()",
        "    body = s3.get_object(Bucket=get_bucket(),",
        "                         Key='models/invoice_anomaly_model.joblib')['Body'].read()",
        "    artifact = joblib.load(io.BytesIO(body))",
        "else:",
        "    artifact = joblib.load('../data/invoice_anomaly_model.joblib')",
        "",
        "pipe = artifact['pipeline']",
        "print('Model loaded.  ROC-AUC =', artifact['metrics']['roc_auc'])",
    ),
    code(
        "# --- Načti nové faktury -------------------------------------------",
        "if os.environ.get('AWS_S3_BUCKET'):",
        "    new_df = load_invoices('invoices_new_batch.csv', s3_client=s3,",
        "                           bucket=get_bucket())",
        "else:",
        "    new_df = load_invoices('../data/invoices_new_batch.csv')",
        "",
        "print(f'Nových faktur ke skórování: {len(new_df)}')",
        "new_df.head()",
    ),
    md(
        "## 3.2 Skórování",
        "",
        "Vlastní inference je jednořádek. **Featury jdou přes stejnou funkci**,",
        "jakou jsme použili při tréninku, s předaným category_index a",
        "vendor_freq_table z train fáze — proto je výsledek deterministický.",
    ),
    code(
        "X_new, _ = build_features(",
        "    new_df,",
        "    category_index=artifact['category_index'],",
        "    vendor_freq_table=pd.Series(artifact['vendor_freq_table']),",
        "    category_stats=pd.DataFrame.from_dict(artifact['category_stats'], orient='index'),",
        ")",
        "",
        "new_df['anomaly_score'] = -pipe.decision_function(X_new)",
        "new_df['is_flagged']    = (pipe.predict(X_new) == -1).astype(int)",
        "",
        "print(f\"Flagnuto: {new_df['is_flagged'].sum()} z {len(new_df)}\")",
    ),
    md(
        "## 3.3 Review queue pro controllera",
        "",
        "Klíčový krok pro byznys: ke každé podezřelé faktuře přidáme **lidsky",
        "čitelný důvod**. Bez něj je model černá skříňka a uživatel ho nepoužije.",
    ),
    code(
        "# Připravíme feature dataframe ve stejném tvaru, jaký dostává explain_row",
        "feat_df = pd.DataFrame(X_new, columns=FEATURE_COLUMNS).astype(float)",
        "feat_df['amount_czk'] = new_df['amount_czk'].values",
        "",
        "queue = new_df.copy()",
        "queue['reason'] = feat_df.apply(explain_row, axis=1)",
        "",
        "queue = (queue",
        "    .sort_values('anomaly_score', ascending=False)",
        "    .loc[:, ['invoice_id', 'vendor', 'amount_czk', 'issued_on',",
        "             'po_number', 'approver', 'anomaly_score', 'reason']]",
        "    .reset_index(drop=True))",
        "",
        "queue.head(15)",
    ),
    md(
        "## 3.4 Export pro controllera",
        "",
        "Výsledek pošleme do bucketu (a do CSV pro Excel). Controller dostane",
        "ráno e-mailem soubor s top 25 položkami ke kontrole.",
    ),
    code(
        "import pandas as pd",
        "from datetime import datetime",
        "",
        "stamp = datetime.now().strftime('%Y-%m-%d')",
        "out_file = f'../data/review_queue_{stamp}.csv'",
        "queue.head(25).to_csv(out_file, index=False)",
        "print('Review queue uložena do:', out_file)",
        "",
        "if os.environ.get('AWS_S3_BUCKET'):",
        "    from invoice_features import write_csv_s3",
        "    write_csv_s3(queue.head(25), s3, get_bucket(),",
        "                 f'review_queue/{stamp}.csv')",
        "    print('A taky do S3:', f'review_queue/{stamp}.csv')",
    ),
    md(
        "## 3.5 Registrace modelu do Model Registry",
        "",
        "Pokud controller potvrdí, že top položky dávají smysl, model přejde do",
        "produkce. To vyřeší Red Hat AI **Model Registry** — promo `Staging` →",
        "`Production` je jeden klik (nebo jeden API call).",
        "",
        "Níže si ukážeme API call. Endpoint registry je k dispozici uvnitř clusteru",
        "na `model-registry-service.<namespace>.svc.cluster.local:8080`.",
    ),
    code(
        "# --- Registrace modelu (volitelné, vyžaduje běžící Model Registry) -",
        "import os, json, requests",
        "",
        "registry_url = os.environ.get('MODEL_REGISTRY_URL')",
        "if registry_url:",
        "    payload = {",
        "        'name': 'invoice-anomaly-detector',",
        "        'description': 'Isolation Forest pro detekci anomálií ve fakturách.',",
        "        'owner': 'controlling@example.com',",
        "    }",
        "    try:",
        "        r = requests.post(f'{registry_url}/api/model_registry/v1alpha3/registered_models',",
        "                          json=payload, timeout=5)",
        "        print('Registry response:', r.status_code)",
        "        print(r.text[:300])",
        "    except Exception as e:",
        "        print('Registry call failed (OK pro lokální spuštění):', e)",
        "else:",
        "    print('MODEL_REGISTRY_URL nenastaveno — přeskakuji.')",
        "    print('Na clusteru: oc get route -n <ns> | grep model-registry')",
    ),
    md(
        "## 3.6 Shrnutí — co tím controller dostal",
        "",
        "1. Ráno mu přistál v inboxu CSV s top 25 podezřelými fakturami.",
        "2. U každé položky vidí **proč** byla označena (kulatá částka, nový",
        "   dodavatel s vysokou částkou, …).",
        "3. Pokud potvrdí, že model funguje, BI tým ho promotne do Production.",
        "4. Když přijde NIS2 auditor, evidence je kompletní:",
        "   `git log` (kód) → MLflow run (parametry + metriky) →",
        "   Model Registry (kdo a kdy schválil) → review_queue/*.csv (výstupy).",
        "",
        "Tohle je celý byznys příběh slide 20, zhmotněný v jednom workbenchi.",
    ),
]

write_notebook("03_score_and_review.ipynb", nb03)
