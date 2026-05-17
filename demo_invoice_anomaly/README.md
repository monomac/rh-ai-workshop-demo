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
| 4 | **Most mezi daty a byznys týmem**        | Notebook 03 produkuje **review queue** s lidsky čitelnými důvody (`explain_row`), **LLM vysvětlovačem** (Phi-4 v `model-test`, OpenAI-kompatibilní API), a **RAG kontextem** přes vendor history (`src/rag_context.py`) — controller dostane konkrétní zdůvodnění "obvykle 27 000 Kč, dnes 1 240 000 Kč = 45× nad medián", ne abstraktní "podezřele vysoká částka". |
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
│   ├── rag_context.py                 # RAG-lite — vendor_history retrieval + CZ prompt format
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

### Co je feature store laicky?

Představ si **sdílený regál s kořením v podnikové kuchyni**. Každý kuchař
(= ML model nebo skripts) má sahnout do toho samého kelímku, když recept
říká "skořice". Bez společného regálu si každý kuchař dělá svoji skořici
po svém — někdo z kůry, někdo z prášku — a výsledné koláče chutnají
pokaždé jinak.

Feature store je přesně tenhle regál pro **featury** — odvozené veličiny,
které model používá k rozhodování. V našem demu to je devět veličin
spočítaných z faktury (logaritmus částky, z-score vůči průměru v kategorii,
příznak víkendu, frekvence dodavatele, …). Bez feature storu by je
trénink a inference počítaly **každý zvlášť**, a jakákoli drobná nuance
v jejich kódu by způsobila, že model **vidí jiné hodnoty v laboratoři
a jiné v produkci** (technický termín: *train-serve skew*).

S feature storem máme:

- **Jeden katalog definic** — kdo featuru vymyslel, kdy, z jakých zdrojů,
  jakou má TTL. Auditor se nemusí ptát "co je `amount_zscore_in_category`?",
  najde si to v UI.
- **Jedny hodnoty** — featuru spočítá *jednou* a v Feastu je uložená.
  Trénink i inference si o ni řekne stejně.
- **Časová pravda** — `event_timestamp` u každé hodnoty znamená, že na
  otázku "co model viděl 10. dubna v 9:43?" existuje deterministická
  odpověď. To je NIS2 / AI Act zlato.
- **Rychlý lookup** v produkci — online store (u nás SQLite, v ostrém
  provozu třeba Redis) odpoví na `get_online_features("INV-2025-08842")`
  v desítkách milisekund. MLServer / KServe inference si může sáhnout
  rovnou tam místo dostávat předpočítaný vektor v requestu.

### Jak to využíváme v tomhle demu?

Workflow je tří-krokový:

1. **Definice v gitu.** `src/feast_repo/{entities,data_sources,feature_views}.py`
   říká *co* jsou naše featury — pojmenování, typy, zdrojový parquet.
   Operator si toto klonuje z `origin/main` a registruje (`feast apply`).
2. **Materializace v notebooku 02 §2.2.** Spočteme 9 featur pro 4 019
   tréninkových faktur, uložíme jako parquet a `feast materialize` to
   nahraje do online storu. Od této chvíle jsou featury *queryable*.
3. **Lookup v notebooku 03 §3.4 (a v produkční inference).**
   `fs.get_online_features(features=[...], entity_rows=[{"invoice_id": x}])`
   vrátí 9 čísel pro danou fakturu v milisekundách. Notebook 03 ukáže,
   že hodnoty se přesně shodují s in-process výpočtem — ne náhodou.

V workshop talk-tracku to obsluhuje slide-20 sloupce *reprodukovatelnost*
(jeden katalog) a *compliance* (časová pravda + audit).

### Co uvidíš v Dashboardu

V levém menu **Develop & train → Feature store** se objeví sedm záložek
(po setupu z `deploy/09-feature-store.yaml`). Co která ukazuje:

| Záložka            | Co je tam                                                                  |
|--------------------|----------------------------------------------------------------------------|
| **Overview**       | Karta projektu `invoice_anomaly` s souhrnem počtů entit / FV / features.   |
| **Entities**       | `invoice_id` — naše entitní klíče. Říká "feature je vždycky vázaná k jedné faktuře". |
| **Data sources**   | `invoice_features_source` — parquet, ze kterého featury pocházejí (s cestou k MinIO). |
| **Datasets**       | Saved snapshoty training datasetů; v našem demu prázdné — featury bereme rovnou z `data sources`. |
| **Features**       | Devět jednotlivých veličin: `log_amount`, `amount_zscore_in_category`, `is_weekend`, `is_holiday`, `missing_po`, `vendor_frequency`, `round_sum_flag`, `days_to_due`, `category_idx`. Klikem na featuru uvidíš metadata, typ, TTL. |
| **Feature views**  | `invoice_features` — view které featury sbalí dohromady (vždycky se publikují jako celek, ne po jedné). |
| **Feature services** | Vyšší abstrakce pro produkci — sada feature views, kterou konkrétní model konzumuje. Pro slide-20 demo to nepotřebujeme; v ostrém nasazení by tam byla služba např. `invoice-fraud-online-v1`. |

Pro workshop demo stačí ukázat **Overview → Entities → Features → Feature
views** a říct: "tohle je ten katalog, který trénink i produkční inference
sdílejí." Auditor pak ví, kde se podívat, který model viděl jaké featury
v jakou dobu.

### Technické pozadí (pro adminy)

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

7. **"A teď ať mi to model vysvětlí česky — a navíc s kontextem historie."** →
   V notebooku 03 §3.5 běží LLM vysvětlovač (`src/explain_with_llm.py`)
   plus RAG-lite nad vendor historií (`src/rag_context.py`). Pro top-25
   podezřelých řádků posílá per-row request na `phi-4-quantizedw8a8-version-1`
   v namespace `model-test` (vLLM, OpenAI-kompatibilní `/v1/chat/completions`).
   Endpoint je v env varech workbenche (`LLM_ENDPOINT`, `LLM_MODEL` —
   viz `deploy/04-workbench.yaml`), takže notebook si ho jen vyzvedne.

   **RAG-lite pattern** — retrieval = `df[df.vendor == X]` přes tréninkový
   set; augment = `format_history_for_prompt` shrne historii do kompaktního
   českého bloku (medián, rozpětí, typická kategorie, ratio bez-PO /
   o-víkendu / kulatých částek, multiple-of-median); generate = Phi-4
   dostane řádek + reason codes + tenhle blok a může citovat konkrétní
   čísla. Bez kontextu LLM píše obecně ("vysoká částka"), s kontextem
   konkrétně ("45× nad medián, vendor obvykle v kategorii facility, ne IT").
   Workshop pointa: tabulkový retrieval má stejnou architekturu jako vector
   RAG, jen jiné retrieval primitivum.

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
  (beat 7, notebook 03 §3.5, Phi-4 W8A8 v `model-test`).
- ~~RAG: nasypat do LLM kontextu i historii faktur od stejného dodavatele~~ —
  **už součást demo** (notebook 03 §3.5 přes `src/rag_context.py`, retrieval
  = vendor filtr nad tréninkovým setem, augment = strukturované shrnutí
  do prompt-u).
- Feedback loop: controller označí false positive → next run snižuje váhu.
- **Vector RAG nad fakturními popisy** — když do faktur přidáme `description`
  (volný text z PDF / ERP), můžeme retrieval rozšířit z `vendor == X` na
  sémantické "podobné popisy v minulosti" (embeddings + pgvector / Qdrant /
  Milvus). To je upgrade na "skutečnou" RAG architekturu, stejný kód-tvar
  v notebooku — jen `retriever.search(query_text, k=8)` místo dataframe
  filtru.
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

## 7 · End-to-end test (dry-run před workshopem)

Po `install.sh` projeď tohle všech 7 beats slide-20 talk-tracku. Trvá
celkem **~15-20 minut**; ověříš tím všechny integrace najednou.

### Pre-flight (1 min, terminál)

```bash
# Workbench Ready + env vars
oc -n rh-ai-workshop get pod invoice-anomaly-wb-0 \
  -o jsonpath='{range .spec.containers[?(@.name=="invoice-anomaly-wb")].env[*]}{.name}={.value}{"\n"}{end}' \
  | grep -E '^(LLM_|FEAST_|MODEL_REGISTRY_URL|AWS_S3)'

# LLM IS Ready (Phi-4)
oc -n model-test get inferenceservice phi-4-quantizedw8a8-version-1

# FeatureStore Ready + visible to Dashboard backend
oc -n rh-ai-workshop get featurestore invoice-anomaly
oc -n redhat-ods-applications exec deploy/rhods-dashboard -c rhods-dashboard -- \
  curl -sS -H "X-Forwarded-Access-Token: $(oc whoami -t)" \
  http://localhost:8080/api/featurestores | python3 -m json.tool

# Model registry reachable + has invoice-anomaly-detector
oc -n rh-ai-workshop exec invoice-anomaly-wb-0 -c invoice-anomaly-wb -- bash -c '
  TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
  curl -sS --cacert /etc/pki/tls/custom-certs/ca-bundle.crt \
    -H "Authorization: Bearer $TOKEN" \
    "$MODEL_REGISTRY_URL/api/model_registry/v1alpha3/registered_models?pageSize=10" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);[print(\"  id=\"+m[\"id\"],m[\"name\"]) for m in d.get(\"items\",[])]"'
```

Všechno musí vrátit Ready / non-empty. Pokud ne → §7 Troubleshooting.

### Beat 1 — Controllerova ranní inbox (30 s)

Otevři **`demo_invoice_anomaly/data/sample_review_queue.csv`** (committed
snapshot). To je výstup, který by controller dostal e-mailem.
**Verify:** 25 řádků, sloupce `vendor`, `amount_czk`, `anomaly_score`, `reason`.

### Beat 2 — Notebook 01 exploration (2 min)

Dashboard → Projects → **RH AI Workshop — Invoice Anomaly Demo** → otevři
workbench. Spusť `notebooks/01_explore_invoices.ipynb` (Run All).
**Verify:** 13 buněk doběhne, distribuce a kategorie se vykreslí.

### Beat 3 — Notebook 02 trénink + Feast publikace (3-5 min)

Spusť `notebooks/02_train_model.ipynb` (Run All). Verify per sekce:

| §   | Verify line                                                                |
|-----|----------------------------------------------------------------------------|
| 2.1 | `Trénovací data: 4 019 faktur`, `Shape: (4019, 9)`                         |
| 2.2 | `feature parquet: 4019 rows × 11 cols`; `feast apply` → No changes; `push 4,019 rows`; online lookup vrátí 9 features |
| 2.3 | trénink hotov                                                              |
| 2.4 | ROC-AUC ≈ 0.957, PR-AUC ~ 0.95                                             |
| 2.5 | `MLflow run logged: <run_id>`                                              |
| 2.6 | `Model uložen i do S3 bucketu`                                             |

### Beat 4 — Audit trail trojúhelník (1 min)

Tři tab refreshe v Dashboardu + jeden terminál v JupyterLab:

- `feast feature-views list` (v `src/feast_repo/`) → vypíše `invoice_features`.
- **MLflow** v levém menu → experiment `invoice-anomaly` → nový run je tam s params/metrics.
- **Models → Model registry → invoice-anomaly-detector** → seznam verzí (pipeline + notebook + nová).
- **Develop & train → Feature store → invoice_anomaly** → Entities `invoice_id`, Data sources, Features (9), Feature views.

### Beat 5 — Live pipeline (3-5 min)

Z laptop terminálu:

```bash
cd "~/Documents/Claude/Projects/Prepare RH AI workshop"
./demo_invoice_anomaly/pipeline/run_pipeline.sh
```

Script vypíše Dashboard URL — klik → **Data Science Pipelines → Runs**.
**Verify:** DAG `ingest → train → evaluate → promote → register`
postupně rozsvícený. Po dokončení Model Registry ukáže novou verzi
`pipeline-<UTC>` s `customProperties.source=data-science-pipelines`.

### Beat 6 — Deploy z Model Registry (1-2 min)

Model Registry → klik na nejnovější `pipeline-<UTC>` → **Deploy model** →
projekt `rh-ai-workshop`. Wizard:

| Step                 | Pole                  | Hodnota                                                       |
|----------------------|-----------------------|---------------------------------------------------------------|
| 1 Model details      | Model location        | Existing connection                                           |
|                      | Connection            | `aws-connection-rhoai-invoices` (MinIO — invoices bucket)     |
|                      | Path                  | `models/invoice-anomaly-detector/<UTC>/` (z `s3_uri` props)   |
|                      | Model type            | **Predictive model**                                          |
| 2 Model deployment   | Framework / runtime   | **sklearn - 1** / MLServer                                    |
| 3 Advanced settings  | (defaulty)            |                                                               |
| 4 Review             | Submit                |                                                               |

**Verify:** Project → Models tab — nová karta Loading → Available za ~30-60 s.
Smoke test:

```bash
PRED=http://<is-name>-predictor.rh-ai-workshop.svc.cluster.local:8080
curl -sS "$PRED/v2/models/<is-name>/infer" -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"x","shape":[1,9],"datatype":"FP64",
                  "data":[[13.5,7.2,1,1,1,1,0,5,3]]}]}'
# → 1 (normal) nebo -1 (anomaly)
```

### Beat 7 — Notebook 03: review queue + Feast online + LLM s RAG (3-4 min)

Spusť `notebooks/03_score_and_review.ipynb` (Run All). Verify per sekce:

| §   | Verify line                                                                  |
|-----|------------------------------------------------------------------------------|
| 3.1 | Model loaded, ROC-AUC = 0.957; new_df 254 faktur                             |
| 3.2 | `Flagnuto: N z 254`                                                          |
| 3.3 | top-15 review queue v output                                                 |
| 3.4 | `Online lookup pro 5 faktur: XX.X ms`, 5× `match=True`                       |
| 3.5 | per-řádek log `[ X/25] Ys vendor → ...`, `vysvětlení` sloupec cituje historii |
| 3.6 | `Review queue uložena do: ../data/review_queue_<datum>.csv` + S3             |
| 3.7 | `Model version vytvořena: iforest-<sha> (id=N)`                              |

---

### Co dělat když něco zaškobrtne

- **Notebook nevidí AWS_S3_BUCKET** → bounce workbench pod (`oc -n rh-ai-workshop delete pod invoice-anomaly-wb-0`).
- **LLM timeout / nedostupný** → `oc -n model-test get inferenceservice phi-4-quantizedw8a8-version-1` (READY=True?). Cold-start ~6-8 min po restartu.
- **Feast UI prázdná** → frontend cache, hard refresh (Cmd+Shift+R) nebo incognito.
- **Deploy stuck na Loading** → `oc -n rh-ai-workshop describe inferenceservice <name>` — typicky storage-initializer connect-refused (KServe annotace na data connection chybí, viz §7).
- **Pipeline `register-model` HTTP 403** → SA `pipeline-runner-dspa` chybí v `deploy/07-model-registry-rbac.yaml`.

---

## 8 · Troubleshooting

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
| `feast get_online_features` vrátí samé `null` u faktur ze setu | Online store nebyl materializován po posledním `feast apply`. Spusť `feast_io.materialize_incremental()` v notebooku — pomocník čte parquet a nahrává řádky přes `FeatureStore.write_to_online_store()`. |
| `feast materialize-incremental` CLI selže s `The input pyarrow table has schema ... with the incorrect columns ['__index_level_0__']` | Známý bug Feast 0.62 LocalComputeEngine — engine zapisuje intermediate výsledky zpátky do offline store v jiném column order, a vlastní schema-validátor to pak odmítne. `feast_io.materialize_incremental()` toto obchází: čte parquet přes pandas a volá `fs.write_to_online_store(fv_name, df)`, který jde rovnou na sqlite. Pokud někdo pustí CLI ručně (`!feast materialize-incremental ...`), narazí. |
| `feast apply` z workbenche selže s `python -m feast: No module named feast.__main__` | Feast SDK nemá `__main__`, má jen setuptools entry point. Volej `feast` binárku z `/opt/app-root/bin/feast`, ne `python -m feast`. `feast_io.feast_cli` to dělá správně. |
| `feast materialize` selže s `ImportError: Install s3fs to access S3` | Workbench image nemá `s3fs`. FileSource v `data_sources.py` proto pointuje na **lokální** parquet (`data/invoice_features.parquet`, vedle feast_repo). Notebook 02 §2.2 píše parquet jak lokálně (pro Feast) tak na S3 (pro audit + cross-env). Pokud potřebuješ s3:// FileSource, doplň `pip install s3fs --break-system-packages` do setupu. |
| Online lookup je rychlý, ale hodnoty se neshodují s in-process compute | Pravděpodobně se změnily lookup tabulky (`category_index` / `vendor_freq_table`) — model byl natrénovaný se starou sadou, novou dávku jsi spočítal s aktuální. Fix: `write_feature_parquet(new_df, artifact=artifact, ...)` v §3.4 notebooku 03 — předáváš modelové lookups, ne re-computed. |
| `feast feature-views list` na clusteru ukazuje 0 FV, i když operator je Ready | Operator klonuje git, ale `runFeastApplyOnInit: false` byl omylem nastavený, nebo feature_views.py má syntax error. `oc logs deploy/feast-invoice-anomaly` ukáže traceback z `feast apply`. |
| Dashboard → Feature Store overview říká "No feature stores are available to users in your organization" i když je FS Ready | Dashboard backend (`backend/dist/routes/api/featurestores/featureStoreUtils.js`) má tři tvrdé podmínky: (1) FS CR musí mít label **`feature-store-ui: enabled`** — `filterEnabledCRDs` vyřazuje vše ostatní; (2) `spec.services.registry.local.server: {}` na CR aby vznikl registry server pod + client ConfigMap s `registry:` blokem; (3) **`server.restAPI: true`** — bez něj operator postaví jen gRPC server a Dashboard dostane `grpc-status: 2 Bad method header`. `deploy/09-feature-store.yaml` v repu všechny tři má. Pokud UI pořád ukazuje "no feature stores" po `oc apply`, je to skoro jistě frontend cache — hard refresh (Cmd+Shift+R) nebo incognito okno. Backend dotaz ověříš: `oc -n redhat-ods-applications exec deploy/rhods-dashboard -c rhods-dashboard -- curl -sS -H "X-Forwarded-Access-Token: $(oc whoami -t)" http://localhost:8080/api/featurestores` — měl by vrátit JSON s tvým FS. |
| Dashboard ukazuje **"Outdated KServe runtime"** u LLM IS | Project-scoped `ServingRuntime` byl naklonován z template, když template ještě měl starší verzi vLLM. Template se mezitím updatoval (`opendatahub.io/runtime-version`), tvoje SR ne. Diff projeď přes `oc -n model-test get servingruntime <name> -o yaml` vs `oc -n redhat-ods-applications get template vllm-cuda-runtime-template -o yaml` — typicky se mění jen `containers[0].image` a anotace. Patch: `oc -n model-test patch servingruntime <name> --type=json -p='[{"op":"replace","path":"/spec/containers/0/image","value":"<new-image>"},{"op":"replace","path":"/metadata/annotations/opendatahub.io~1runtime-version","value":"v<new>"}]'`. Deployment udělá rolling update — starý pod běží dokud nový nepasuje readiness, takže LLM endpoint zůstane funkční po celou dobu cold pullu (~5-7 min). |
