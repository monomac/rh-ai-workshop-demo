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
| 3 | **Reprodukovatelnost a sdílení**         | `src/invoice_features.py` (1 funkce pro train i score) + MLflow run ID + Git-friendly .ipynb + **Feast feature store** (`src/feast_repo/`) jako platformně vynucený single-source-of-truth pro definice featur — git je zdroj, RHOAI feastoperator zaregistruje. |
| 4 | **Most mezi daty a byznys týmem**        | Notebook 03 produkuje **review queue** s lidsky čitelnými důvody (`explain_row`) a navíc **LLM vysvětlovačem** (Phi-4 v `model-test`, OpenAI-kompatibilní API) — controller dostane jedno-větné česky znějící zdůvodnění u každé top-25 podezřelé faktury. |
| 5 | **AI quickstart pattern**                | Celé repo je quickstartovatelná šablona: `deploy/install.sh` postaví prostředí (vč. MLServer ServingRuntime), `pipeline/run_pipeline.sh` spustí trénink+registraci jedním příkazem, "Deploy" v Model Registry vystaví model jedním kliknutím. |
| 6 | **Auditovatelnost a compliance**         | Feast registry (definice featur, kdo a kdy) → MLflow run (notebook trénink) → Model Registry verze (kdo schválil) → DS Pipelines run (kdy běželo automaticky) → KServe InferenceService (kdy a kdo deploynul do produkce) → Feast online store (snapshot featur k danému timestampu) → versionovaný S3 klíč (immutable bytes). Plná evidence pro NIS2 / AI Act, včetně "co model viděl, když rozhodoval". |

---

## 2 · Co je v repu

```
demo_invoice_anomaly/
├── README.md                          # tenhle soubor
├── data/                              # synthetic data (commited do git pro reprodukci)
│   ├── invoices.csv                   # 4 019 řádků, 3 % anomálií (training set)
│   ├── invoices_new_batch.csv         # 254 řádků (fresh batch ke skórování)
│   └── sample_review_queue.csv        # referenční výstup notebooku 03 (top 25 podezřelých)
├── requirements.txt                   # prázdné — vše je v image (vč. mlflow), viz § 3
├── src/
│   ├── generate_invoices.py           # generátor — pokud chceš jiný objem nebo seed
│   ├── invoice_features.py            # SDÍLENÉ featury (train == score)
│   ├── explain_with_llm.py            # LLM vysvětlovač — OpenAI-kompatibilní klient (stdlib)
│   ├── feast_io.py                    # Feast helpers — apply / materialize / parquet write
│   ├── feast_repo/                    # Feast project (čte feastoperator i workbench)
│   │   ├── feature_store.yaml         # local provider, sqlite online, file offline
│   │   ├── entities.py                # invoice entity
│   │   ├── data_sources.py            # FileSource → MinIO parquet
│   │   └── feature_views.py           # 9-feature FeatureView
│   └── build_notebooks.py             # source-of-truth notebooků (programaticky generované)
├── notebooks/
│   ├── 01_explore_invoices.ipynb      # controller POV — průzkum dat
│   ├── 02_train_model.ipynb           # trénink Isolation Forest + MLflow
│   └── 03_score_and_review.ipynb      # skórování + review queue + Model Registry
├── pipeline/
│   ├── invoice_anomaly_pipeline.py    # KFP v2 DSL — zdroj pravdy
│   ├── invoice_anomaly_pipeline.yaml  # zkompilovaný pipeline (commit i tady)
│   └── run_pipeline.sh                # one-command live trigger pro workshop
└── deploy/                            # OpenShift manifesty pro sa-ai cluster
    ├── 01-namespace.yaml
    ├── 02-minio.yaml
    ├── 03-data-connection.yaml         # S3 connection secret (+ KServe annotace pro "Deploy")
    ├── 04-workbench.yaml               # workbench Notebook CR + SA + PVC + env (vč. MODEL_REGISTRY_URL)
    ├── 05-dspa.yaml
    ├── 06-bootstrap-data.yaml
    ├── 07-model-registry-rbac.yaml     # cross-ns RoleBinding → registry-user-elos-model-registry
    ├── 08-serving-runtime.yaml         # MLServer ServingRuntime — povolí "Deploy" z Model Registry
    ├── 09-feature-store.yaml           # Feast FeatureStore CR (operator-managed, git-clones src/feast_repo)
    └── install.sh
```

---

## 3 · MLflow přímo z platformy (RHOAI 3.4)

MLflow se v `demo_invoice_anomaly` **nic neinstaluje**. RHOAI 3.4 ho:

- má v Standard DS Notebook image (`mlflow 3.10.1+rhaiv.3`, Red Hat build),
- provozuje jako platformní službu přes DSC komponentu `mlflowoperator`
  (kontrola: `oc get dsc -A -o jsonpath='{.items[*].spec.components.mlflowoperator.managementState}'`),
- automaticky workbench podu předá `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_AUTH=kubernetes-namespaced`,
  `MLFLOW_K8S_INTEGRATION=true` a namountuje SA token z `rh-ai-workshop` namespace.

Auth model: server validuje práva přes `SelfSubjectAccessReview`, **namespace = workspace**.
RBAC (`<wb>-mlflow` → ClusterRole `mlflow-operator-mlflow-integration` + `system:auth-delegator`)
zakládá operator sám, jakmile workbench vznikne. Detaily viz
[KCS 7136121](https://access.redhat.com/articles/7136121).

## 3a · Lokální spuštění (suchý běh před workshopem)

```bash
# Lokální závislosti (mimo cluster nutno doinstalovat z PyPI):
pip install pandas numpy scikit-learn matplotlib 'mlflow>=2.16,<3' joblib jupyter
cd demo_invoice_anomaly
python src/generate_invoices.py            # vygeneruje data/invoices.csv
jupyter lab notebooks/                     # otevři 01 → 02 → 03 postupně
```

Notebooky se spustí beze změny i bez clusteru — detekují, že `AWS_S3_BUCKET` není
v env, načtou data z lokálního `../data/`, a MLflow zaloguje do `file:./mlruns`.

**Ověřeno:** všechny tři notebooky proběhnou end-to-end na čisté instalaci
Python 3.10 + balíčky výše. Trénink dává **ROC-AUC ≈ 0.96** a **94 % precision
v top-50** podezřelých faktur na synthetic datech.

## 3b · Feast feature store

Feast je v `demo_invoice_anomaly` druhou platformní službou vedle MLflow —
RHOAI 3.4 ho provozuje přes DSC komponentu `feastoperator` (`Managed`).
Operator stahuje feature definice z gitu, registruje je a vystavuje online
endpoint:

```
git (src/feast_repo/) ──┐
                        ├─→ feastoperator pod (clone + feast apply)
deploy/09-feature-store ┘        │
                                 ├─→ /feast-data/registry.db    (definice)
                                 ├─→ /feast-data/online_store.db (SQLite)
                                 └─→ Service feast-invoice-anomaly-online:443
                                       └─→ Client ConfigMap mounted in workbench
```

Co kde žije:

- `src/feast_repo/feature_store.yaml` — Feast config (project `invoice_anomaly`,
  local provider, sqlite online, file offline). Čte to operator i workbench.
- `src/feast_repo/{entities,data_sources,feature_views}.py` — Python kód,
  který Feast `apply` převede do registry. Source of truth pro featury.
- `src/feast_io.py` — workbench/pipeline helper: `write_feature_parquet`,
  `apply_feature_definitions`, `materialize_incremental`, `open_local_store`.
- `deploy/09-feature-store.yaml` — FeatureStore CR. Pointuje
  `feastProjectDir.git.url` na tenhle workshop repo, `featureRepoPath` na
  `demo_invoice_anomaly/src/feast_repo`.

**Ordering caveat:** operator se snaží git-clonovat při bootu CR. Feature
definitions musí být na `origin/main` **dřív**, než se aplikuje
`09-feature-store.yaml`. Pokud aplikuješ CR proti repu, který ještě nemá
`src/feast_repo/`, operator pod uvízne v `CrashLoopBackOff` s
`fatal: path 'demo_invoice_anomaly/src/feast_repo' does not exist`. Fix:
push feature definitions na main, pak `oc delete pod -l app.kubernetes.io/instance=invoice-anomaly`
ať operator re-init.

Workbench čte client config z mountu `/opt/app-root/src/feast-client/feature_store.yaml`,
ale notebook používá *vlastní* lokální Feast SDK config (`src/feast_repo/`),
aby šlo z workbench dělat `apply` a `materialize` bez závislosti na shared
PVC s operator podem. Online endpoint operatoru je `Co dál` — pro MLServer
inference v produkci.

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

# 4. nahraj pipeline + spusť první běh (one-command)
./demo_invoice_anomaly/pipeline/run_pipeline.sh
# script si sám:
#   - zkompiluje yaml z .py (pokud .py je novější)
#   - založí pipeline `invoice-anomaly` (nebo k ní přidá novou verzi)
#   - vytvoří run s display name `invoice-anomaly-<UTC-timestamp>`
#   - vypíše URL na běh v Dashboard
# Alternativně manuálně: Dashboard → Data Science Pipelines → Import pipeline.
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
   Model Registry (kdo schválil promo do Production) → **Feast registry**
   (`feast feature-views list` v terminálu workbenche, nebo přímo
   `oc -n rh-ai-workshop get featurestore invoice-anomaly -o yaml` → spec
   ukazuje, *odkud* z gitu featury přišly a *kdy* byly registrované).
   Pro auditora: "podívej se, model viděl těchhle 9 featur, definované
   tímhle git commitem, vypočítané z tohoto event_timestampu" —
   Feast point-in-time semantika to dělá deterministicky.
5. **"A když to chceme každou noc automaticky — pojďme to spustit teď"** →
   přepni do terminálu a v jednom příkazu spusť pipeline naživo:

   ```bash
   ./demo_invoice_anomaly/pipeline/run_pipeline.sh
   ```

   Script vypíše Dashboard URL — klikni na ni a přepni se na **DS Pipelines →
   Runs**. Audience uvidí, jak se DAG `ingest → train → evaluate →
   promote-if-passing → register` postupně rozsvěcuje:

   - **ingest-invoices** — natáhne `invoices.csv` z MinIO (často cached,
     skip pod ~2 s — řekni, že KFP cache je *feature*, ne *bug*: identický
     vstup = neopakuj práci).
   - **train-model** — Isolation Forest, vrátí ROC-AUC jako parametr.
   - **evaluate-model** — gate, propustí pouze ROC-AUC ≥ 0.85.
   - **register-model** — upload nového `.joblib` do `s3://.../models/<version>/`
     a POST do **elos-model-registry** jako nová `pipeline-<UTC>` verze.
     Auth: pipeline SA má bearer token přes
     `deploy/07-model-registry-rbac.yaml` (RoleBinding na
     `registry-user-elos-model-registry` v `rhoai-model-registries`).

   Celý běh trvá ~2–3 minuty (první cold run cca 5 minut kvůli image pull).
   Když doběhne, přepni do **Models** → `invoice-anomaly-detector` a ukaž
   novou `pipeline-<timestamp>` verzi se `source=data-science-pipelines` a
   `s3_uri` v custom properties. **Tohle je celý compliance příběh:**
   git commit (kód) → MLflow run (notebook trénink) → pipeline run
   (automatizace) → Model Registry verze (kdo, kdy, odkud) → S3 artifact
   (immutable bytes).

   > Když chce někdo z publika vidět **kód** té automatizace, otevři
   > `pipeline/invoice_anomaly_pipeline.py` — je krátký a doslova ten samý
   > feature engineering jako v notebooku, jen zabalený do KFP DSL.

6. **"A teď ho nakliknu do produkce."** → V Dashboardu přepni do
   **Model Registry → invoice-anomaly-detector**, klikni na novou verzi
   `pipeline-<UTC>`, vpravo nahoře **Deploy → rh-ai-workshop**, ve formuláři
   vyber:

   - Model serving runtime: **MLServer ServingRuntime for KServe**
     (předem připravený z `deploy/08-serving-runtime.yaml`)
   - Connection: **MinIO — invoices bucket** (`aws-connection-rhoai-invoices`,
     má v sobě KServe annotace `serving.kserve.io/s3-endpoint`,
     `s3-usehttps=0`, `s3-verifyssl=0`).
   - Path: vyplní se z `s3_uri` customProperty té verze.

   Submit → přepni do **rh-ai-workshop → Models**. Audience uvidí novou kartu
   modelu s "Loading" → "Available" za ~30–60 sekund. Pod kartou je internal
   endpoint a kdo má roli `view` v projektu, může model rovnou volat:

   ```bash
   # z terminálu workbenche, nebo z laptopu přes oc port-forward
   PRED=http://<inference-service>-predictor.rh-ai-workshop.svc.cluster.local:8080
   curl -sS "$PRED/v2/models/<name>/infer" -H "Content-Type: application/json" \
     -d '{"inputs":[{"name":"x","shape":[1,8],"datatype":"FP64",
                     "data":[[13.5,7.2,1,1,1,1,0,5]]}]}'
   # → výstup: -1 (anomálie) nebo 1 (normální) podle IsolationForest konvence
   ```

   **Co se právě stalo** (audit pohled): controller schválil verzi → BI tým
   ji nakliknul z registru → KServe vytvořil `InferenceService` →
   storage-initializer stáhl `model.joblib` z S3 (přes data-connection,
   která má KServe annotace) → MLServer ho načetl a vystavil REST endpoint
   v Open Inference Protocol v2. **Nikdo nepsal řádku Python** mezi
   "model je zaregistrovaný" a "model je v produkci".

7. **"A teď ať mi to model vysvětlí česky."** → V notebooku 03 §3.4
   běží LLM vysvětlovač (`src/explain_with_llm.py`). Pro top-25 podezřelých
   řádků posílá per-row request na `phi-4-quantizedw8a8-version-1` v
   namespace `model-test` (vLLM, OpenAI-kompatibilní `/v1/chat/completions`).
   Endpoint je v env varech workbenche (`LLM_ENDPOINT`, `LLM_MODEL` —
   viz `deploy/04-workbench.yaml`), takže notebook si ho jen vyzvedne.

   - Systémový prompt v češtině zakazuje halucinace mimo dodaná data:
     LLM dostane řádek faktury + reason codes + anomaly score, vrací
     2-3 věty „proč je to podezřelé". Dořeší se tím poslední slabina
     review queue — `reason` sloupec je věcný, ale suchý.
   - Latence na A10G ~3-5 s/řádek, celkem ~1-2 min na 25 položek.
     Apertus-8B na stejném GPU je ~2× rychlejší (nižší parametry,
     fp8 quant), případně použij jako fallback (přepiš env vars
     v workbenchi).
   - Pokud LLM endpoint nepojede, funkce vrátí fallback string a
     notebook bez chyby pokračuje na §3.5 (export) — review queue
     odejde jen s `reason`, bez `vysvětlení`.

   **Audit-pohled na tenhle krok:** všechny prompty + odpovědi jdou
   pod RBAC stejné SA jako zbytek workbenche; vLLM zalogovává request
   v `model-test/phi-4-...-predictor`. Pokud auditor chce vidět "proč
   tohle model říká", má v ruce vstup (řádek faktury + reason codes)
   i výstup (CSV se sloupcem `vysvětlení`).

---

## 6 · Co dál (out-of-scope pro tenhle workshop)

- ~~Online inference přes KServe / Model Serving~~ — **už součást demo** (beat 6,
  ServingRuntime + InferenceService přes Dashboard "Deploy"). To, co zůstává
  out-of-scope:
  - Production-grade ingress / mTLS / rate-limiting před prediktorem.
  - Multi-model serving přes ModelMesh (pro shop s tisíci modely).
  - A/B test mezi novou a starou verzí přes Knative traffic-splitting.
- ~~LLM-vysvětlovač "Proč právě tohle podezření?"~~ — **už součást demo**
  (beat 7, notebook 03 §3.4, Phi-4 W8A8 v `model-test`).
- Feedback loop: controller označí false positive → next run snižuje váhu.
- RAG: nasypat do LLM kontextu i historii faktur od stejného dodavatele,
  aby vysvětlení rovnou cituovalo "pětkrát jste platili pod 50 k, teď 1 M".
- **Pipeline (KFP) integrace s Feast** — `ingest_invoices` komponenta čte
  z Feast historical join místo z CSV, `register_model` taky materializuje
  do online store. Notebooks 02/03 už to dělají; pipeline je deferred,
  protože vyžaduje `feast` SDK v pipeline images + RBAC mount client
  ConfigMap do pipeline pods. Plánováno na další session.
- **MLServer ↔ Feast online endpoint** — produkční inference (beat 6
  deploy z Model Registry) by si v ostrém provozu sahala pro featury do
  Feast `feast-invoice-anomaly-online:443` přes invoice_id, ne dostávala
  pre-computed vektory. Vyžaduje custom MLServer model wrapper.

---

## 7 · Troubleshooting

| Problém                                                | Řešení                                                                                  |
|--------------------------------------------------------|------------------------------------------------------------------------------------------|
| Notebook nevidí `AWS_S3_BUCKET`                        | Data Connection nepřipojená k workbenchi. Dashboard → workbench → Attach connection.    |
| Pipeline failuje s "no access to S3"                   | DSPA má jinou `s3CredentialsSecret`. Sjednoť s `aws-connection-rhoai-invoices`.         |
| `pip install mlflow` v notebooku selže (`No matching distribution`) | Workbench používá interní RH PyPI mirror, kde je jen RH-build `mlflow 3.10.x+rhaiv.*`. Image už mlflow obsahuje — žádný `pip install` v notebooku **není potřeba**. |
| Notebook 03 §3.5 → `NameResolutionError` / `Failed to resolve model-registry-service.<ns>.svc...` | Notebook CR má starý `MODEL_REGISTRY_URL`. Reapply `deploy/04-workbench.yaml` (nový pointuje na externí route registru) a smaž pod, ať si vezme nové env. |
| Notebook 03 §3.5 → HTTP 403 z registru                  | SA nemá RBAC na `services/elos-model-registry`. Apply `deploy/07-model-registry-rbac.yaml`. |
| Notebook 03 §3.5 → HTTP 422 "registeredModelId is zero value" | API v1alpha3 vyžaduje `registeredModelId` v body i v URL cestě verze. Notebook to už dělá; pokud editujete buňku, nezapomeňte na to. |
| MLflow run nezaloguje                                  | Operator nepřipojil env. Zkontroluj `oc get dsc -A` → `mlflowoperator: Managed`; `oc get mlflow -A` → instance `Available`; restartni workbench pod (RoleBindings + env injektuje operator při startu podu). |
| `oc apply` na DSPA selže s `no matches for kind …`     | Komponenta `datasciencepipelines` v DSC je `Removed`. Změň na `Managed`.                 |
| `oc apply` na Notebook selže s `no matches for kind …` | Komponenta `workbenches` v DSC je `Removed`. Změň na `Managed`.                          |
| Pipeline `ingest-invoices` umírá s `NoCredentialsError: Unable to locate credentials` | Komponentě chybí AWS env vars. Pipeline DSL musí mít `kubernetes.use_secret_as_env(task, secret_name="aws-connection-rhoai-invoices", ...)` — viz `pipeline/invoice_anomaly_pipeline.py`. |
| Pipeline `pip install` umírá s `No matching distribution found for boto3==1.34.*` | RHOAI interní PyPI mirror má jen RH-blessed verze (boto3 1.35+, ne 1.34). `packages_to_install` musí používat floor-pins (`boto3>=1.35,<2.0`), ne `==1.34.*`. |
| Pipeline `register-model` → HTTP 403 / "forbidden" z Model Registry | `pipeline-runner-dspa` SA chybí v RoleBinding `invoice-anomaly-wb-registry-user`. Apply aktuální `deploy/07-model-registry-rbac.yaml` — má dva subjekty (wb + pipeline-runner). |
| Pipeline `evaluate-model` umírá s `FileNotFoundError: ...out_metrics` | KFP v2 `Output[Metrics]` nepublikuje soubor na `in_metrics.path`; metriky žijí v MLMD metadatech, ne ve workspace artifactu. Předávej skóre jako `float` parameter mezi train ↔ evaluate, ne přes `Input[Metrics]`. |
| `run_pipeline.sh` se přihlásí, ale `curl ... /pipelines/upload?name=invoice-anomaly&description=něco česky` vrátí HTTP 400 z haproxy | Unicode v query stringu rozhoupe haproxy. Script proto popis nedává do query (description ukládá KFP z pipeline `description` v DSL). |
| **Deploy z Model Registry** vytvoří InferenceService, ale predikce vrací HTTP 500 `'dict' object has no attribute 'predict'` | Starší verze pipeline ukládaly do S3 dict bundle (`{"pipeline": ..., "feature_columns": ...}`). MLServer's sklearn runtime volá `.predict()` přímo na výsledku `joblib.load()`, takže potřebuje *bare* sklearn estimator. Aktuální `register_model` zapisuje vedle sebe `model.joblib` (sklearn-only) a `bundle.joblib` (dict, pro notebooky). Pokud máš staré verze, spusť backfill: download `model.joblib`, `joblib.load(...)["pipeline"]`, upload zpátky. |
| **Deploy z Model Registry** se zasekne na "Loading" / storage-initializer hází `dial tcp: connection refused` | `aws-connection-rhoai-invoices` secret nemá KServe annotace. Apply aktuální `deploy/03-data-connection.yaml` — má `serving.kserve.io/s3-endpoint`, `s3-usehttps=0`, `s3-verifyssl=0`. |
| V Dashboard `Models` tab je prázdno i po Deploy | Project je nesprávně označen jako `modelmesh-enabled` (ModelMesh, ne KServe). U RHOAI 3.4 self-managed by `kserve` v DSC mělo být `Managed` a `modelmeshserving` neenabled. `oc get dsc -o yaml \| grep -A1 'kserve:'` ověří. |
| Notebook 03 §3.4 → `Could not resolve host: phi-4-...predictor.model-test.svc.cluster.local` | LLM IS není READY → headless Service nemá endpoints → DNS nevrátí adresu. `oc -n model-test get pods,is` — počkej, až `predictor` pod je Ready. Workshop tip: nech LLM předhřátý před demem (initial cold pull modelcar 1.5 je ~15 GB, ~5 minut). |
| Notebook 03 §3.4 → connection refused na `:80` | Headless Service nemá proxy na `:80`. Klient musí mluvit na port `8080` (pod listening port). `LLM_ENDPOINT` v `deploy/04-workbench.yaml` to už má; pokud editujete, **musí končit `:8080/v1`**, ne `/v1`. |
| Notebook 03 §3.4 → CZ výstup zní lámaně / kazí diakritiku | Model nemá oficiální CS support. Apertus-8B i Phi-4 14B mají CS v supported language listu; Granite 3.x a Llama 3.1 ne. Přepiš `LLM_MODEL` / `LLM_ENDPOINT` na supported model. |
| Notebook 03 §3.4 trvá víc než 5 minut | Buď nemá GPU node (předpoklad: A10G/L4 24 GB VRAM, viz `g5.xlarge` nebo `g6.2xlarge` MachineSet), nebo vLLM běží na CPU runtime → 5-15 tok/s. Zkontroluj `oc -n model-test describe pod <predictor> \| grep nvidia.com/gpu` — request musí mít `1`. |
| FeatureStore CR pod v `CrashLoopBackOff` s `fatal: path '…/feast_repo' does not exist` | Operator klonuje git ref v `feastProjectDir.git`. Soubory v `demo_invoice_anomaly/src/feast_repo/` musí být na origin/main předtím, než aplikuješ CR. Fix: commit + push featur, pak `oc -n rh-ai-workshop delete pod -l app.kubernetes.io/instance=invoice-anomaly` ať operator re-init z aktuálního ref. |
| `feast apply` z workbenche selže s `botocore.exceptions.NoCredentialsError` | FileSource v `data_sources.py` čte z `s3://...` přes pyarrow/s3fs. Workbench potřebuje AWS_* z `aws-connection-rhoai-invoices` (envFrom secretRef už je v `04-workbench.yaml`). Restart pod, případně ověř `oc exec invoice-anomaly-wb-0 -- env \| grep AWS_`. |
| `feast get_online_features` vrátí samé `null` u faktur ze setu | Online store nebyl materializován po posledním `feast apply`, nebo materialize watermark přeskočil novou dávku. Spusť `feast_io.materialize_incremental()` v notebooku — Feast posune watermark a nahraje features z parquet do SQLite. |
| Online lookup je rychlý, ale hodnoty se neshodují s in-process compute | Pravděpodobně se změnily lookup tabulky (`category_index` / `vendor_freq_table`) — model byl natrénovaný se starou sadou, novou dávku jsi spočítal s aktuální. Fix: `write_feature_parquet(new_df, artifact=artifact, ...)` v §3.4 notebooku 03 — předáváš modelové lookups, ne re-computed. |
| `feast feature-views list` na clusteru ukazuje 0 FV, i když operator je Ready | Operator klonuje git, ale `runFeastApplyOnInit: false` byl omylem nastavený, nebo feature_views.py má syntax error. `oc logs deploy/feast-invoice-anomaly` ukáže traceback z `feast apply`. |
| Dashboard ukazuje **"Outdated KServe runtime"** u LLM IS | Project-scoped `ServingRuntime` byl naklonován z template, když template ještě měl starší verzi vLLM. Template se mezitím updatoval (`opendatahub.io/runtime-version`), tvoje SR ne. Diff projeď přes `oc -n model-test get servingruntime <name> -o yaml` vs `oc -n redhat-ods-applications get template vllm-cuda-runtime-template -o yaml` — typicky se mění jen `containers[0].image` a anotace. Patch: `oc -n model-test patch servingruntime <name> --type=json -p='[{"op":"replace","path":"/spec/containers/0/image","value":"<new-image>"},{"op":"replace","path":"/metadata/annotations/opendatahub.io~1runtime-version","value":"v<new>"}]'`. Deployment udělá rolling update — starý pod běží dokud nový nepasuje readiness, takže LLM endpoint zůstane funkční po celou dobu cold pullu (~5-7 min). |
