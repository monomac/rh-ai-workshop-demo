# Invoice Anomaly Demo — slide 20 walkthrough

**Workshop scénář:** *Detekce anomálií ve fakturách (pohled controllera)*
**Cíl:** Ukázat na jednom funkčním příkladu všech šest bodů ze slide 20
("Co z platformy má byznys") pomocí Red Hat AI 3.4.

---

## 1 · Mapování na slide 20

| # | Bod ze slide 20                          | Kde to v demo uvidíme                                                                              |
|---|------------------------------------------|----------------------------------------------------------------------------------------------------|
| 1 | **Rychlejší prototypování**              | `notebooks/01_explore_invoices.ipynb` — od nuly k první vizualizaci v 5 buňkách.                   |
| 2 | **Self-service pro analytiky**           | Notebook 02 má parametry v top-level buňce; doménový expert je tweakuje bez vývojáře.              |
| 3 | **Reprodukovatelnost a sdílení**         | `src/invoice_features.py` (1 funkce pro train i score) + MLflow run ID + Git-friendly .ipynb.      |
| 4 | **Most mezi daty a byznys týmem**        | Notebook 03 produkuje **review queue** s lidsky čitelnými důvody (`explain_row`).                  |
| 5 | **AI quickstart pattern**                | Celé repo je quickstartovatelná šablona — `install.sh`, `pipeline/`, `deploy/`. Stáhneš → spustíš. |
| 6 | **Auditovatelnost a compliance**         | MLflow → Model Registry → DS Pipeline + verzovaný S3 klíč. Plná evidence pro NIS2 / AI Act.        |

---

## 2 · Co je v repu

```
demo_invoice_anomaly/
├── README.md                          # tenhle soubor
├── data/                              # synthetic data (commited do git pro reprodukci)
│   ├── invoices.csv                   # 4 019 řádků, 3 % anomálií (training set)
│   ├── invoices_new_batch.csv         # 254 řádků (fresh batch ke skórování)
│   └── sample_review_queue.csv        # referenční výstup notebooku 03 (top 25 podezřelých)
├── src/
│   ├── generate_invoices.py           # generátor — pokud chceš jiný objem nebo seed
│   ├── invoice_features.py            # SDÍLENÉ featury (train == score)
│   └── build_notebooks.py             # source-of-truth notebooků (programaticky generované)
├── notebooks/
│   ├── 01_explore_invoices.ipynb      # controller POV — průzkum dat
│   ├── 02_train_model.ipynb           # trénink Isolation Forest + MLflow
│   └── 03_score_and_review.ipynb      # skórování + review queue + Model Registry
├── pipeline/
│   ├── invoice_anomaly_pipeline.py    # KFP v2 DSL
│   └── invoice_anomaly_pipeline.yaml  # zkompilovaný pipeline (upload do RHOAI)
└── deploy/                            # OpenShift manifesty pro sa-ai cluster
    ├── 01-namespace.yaml
    ├── 02-minio.yaml
    ├── 03-data-connection.yaml
    ├── 04-workbench.yaml
    ├── 05-dspa.yaml
    ├── 06-bootstrap-data.yaml
    └── install.sh
```

---

## 3 · Lokální spuštění (suchý běh před workshopem)

```bash
pip install pandas numpy scikit-learn matplotlib mlflow joblib jupyter
cd demo_invoice_anomaly
python src/generate_invoices.py            # vygeneruje data/invoices.csv
jupyter lab notebooks/                     # otevři 01 → 02 → 03 postupně
```

Notebooky se spustí beze změny bez clusteru — detekují, že `AWS_S3_BUCKET` není
v env, a načtou data z lokálního adresáře `../data/`.

**Ověřeno:** všechny tři notebooky proběhnou end-to-end na čisté instalaci
Python 3.10 + balíčky výše. Trénink dává **ROC-AUC ≈ 0.96** a **94 % precision
v top-50** podezřelých faktur na synthetic datech.

---

## 4 · Nasazení na sa-ai cluster (RHOAI 3.4 self-managed)

> Vychází z toho, že DSC/DSCI jsou nasazené (v2 API) a Model Registry komponenta
> je `Managed`. Pokud ne, viz `UPGRADE_HANDOVER.md` v root projektu.

```bash
# 1. přihlas se na cluster jako `sp`
oc login --server=https://api.sa-ai.<base-domain>:6443 -u sp

# 2. spusť installer (vytvoří namespace, MinIO, DSPA, workbench, nahraje data)
./demo_invoice_anomaly/deploy/install.sh

# 3. v RHOAI Dashboard → "RH AI Workshop — Invoice Anomaly Demo"
#    - klikni Open na workbench
#    - clone tohle repo dovnitř (`git clone …` v terminálu workbenche)
#    - otevři notebooks/ a spusť 01 → 02 → 03

# 4. nahraj pipeline
oc -n rh-ai-workshop apply -f - <<<"$(cat pipeline/invoice_anomaly_pipeline.yaml)"
# nebo přes Dashboard → Data Science Pipelines → Import pipeline
```

---

## 5 · Workshop talk-track (rychlá osnova)

Když ukazuješ slide 20 a přepínáš do RHOAI:

1. **"Tohle je co business sponzor uvidí ráno"** → otevři `data/sample_review_queue.csv`
   (committed referenční výstup z notebooku 03). Top 25 podezřelých faktur
   s důvody. Při živém běhu notebook produkuje `data/review_queue_<datum>.csv`
   (gitignored, per-run output).
2. **"A takhle k tomu controller došel"** → workbench, notebook 01.
   Žádný IT ticket, žádná infra, jen Data Connection a 5 buněk.
3. **"Když chce model dotrénovat, parametry jsou tady"** → notebook 02,
   PARAMETRY buňka. Doménový expert ji upraví.
4. **"Audit trail pro NIS2"** → MLflow UI (run params + metrics) →
   Model Registry (kdo schválil promo do Production).
5. **"A když to chceme každou noc automaticky"** → DS Pipelines tab,
   ukaž graf z `invoice_anomaly_pipeline.yaml`.

---

## 6 · Co dál (out-of-scope pro tenhle workshop)

- Online inference přes KServe / Model Serving (jen joblib na disku stačí).
- LLM-vysvětlovač "Proč právě tohle podezření?" (Granite + RAG nad fakturami).
- Feedback loop: controller označí false positive → next run snižuje váhu.

---

## 7 · Troubleshooting

| Problém                                                | Řešení                                                                                  |
|--------------------------------------------------------|------------------------------------------------------------------------------------------|
| Notebook nevidí `AWS_S3_BUCKET`                        | Data Connection nepřipojená k workbenchi. Dashboard → workbench → Attach connection.    |
| Pipeline failuje s "no access to S3"                   | DSPA má jinou `s3CredentialsSecret`. Sjednoť s `aws-connection-rhoai-invoices`.         |
| MLflow run nezaloguje                                  | `MLFLOW_TRACKING_URI` v env workbenche nesedí, nebo MLflow pod neběží — zkontroluj svc. |
| `oc apply` na DSPA selže s `no matches for kind …`     | Komponenta `datasciencepipelines` v DSC je `Removed`. Změň na `Managed`.                 |
| `oc apply` na Notebook selže s `no matches for kind …` | Komponenta `workbenches` v DSC je `Removed`. Změň na `Managed`.                          |
