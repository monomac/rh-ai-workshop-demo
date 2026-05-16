# RHOAI 3.4.0 Upgrade — Session Handover (updated 2026-05-15)

## Status: operator upgraded, **dashboard/workbenches/sparkoperator blocked on cluster capacity**

Previous attempt got stuck on egress / a wrong assumption about the install state.
This session completed the operator-side upgrade. Remaining work is purely
cluster-capacity related and will resolve once the user scales the ROSA nodepool.

## What's now true on the cluster (`sa-ai`)

- **Operator**: `rhods-operator.3.4.0` (GA), CSV `Succeeded`, Subscription on
  channel `stable-3.4`, source `redhat-operators`. 3 pods Running.
- **DSC**: `default-dsc` applied via the **v2 API** (the v1 validator in 3.4
  rejects `managementState: Managed` — that's a quirk to remember; use v2).
  Components set to `Managed`: aipipelines, dashboard, feastoperator, kserve
  (Serverless + OpenshiftDefaultIngress), llamastackoperator, mlflowoperator,
  modelregistry, ray, sparkoperator, trainer, trainingoperator, trustyai,
  workbenches. `kueue` is `Unmanaged` (3.4 expects the standalone Red Hat
  build of Kueue operator, which is already installed and running).
  `codeflare` and `modelmeshserving` no longer exist as components in v2.
- **DSCI**: `default-dsci` survived the soft reinstall, still Ready.
- **Components Ready (12/14)**: AIPipelines, FeastOperator, Kserve, Kueue,
  LlamaStackOperator, MLflowOperator, ModelController, ModelRegistry, Ray,
  Trainer, TrainingOperator, TrustyAI.
- **Components blocked on scheduling**: Dashboard (0/1 deployments),
  Workbenches (0/2 deployments), SparkOperator (1/2 deployments).
- **User workloads** in `model-test` and `rhoai-model-registries` are intact.
  `rag-blueprint` was already in Terminating state since 2026-04-21 (stuck
  finalizer); the new operator unblocked it and the namespace finished
  deleting — not data loss, just cleanup of a long-stuck deletion.

## Pending: ROSA nodepool scale-up

All 5 worker nodes are 90–99% CPU-requested. The 3.4 `rhods-dashboard` pod
asks for ~3.6 CPU per replica (9 UI containers: Dashboard, ModelRegistry UI,
GenAI UI, MaaS UI, MLflow UI, EvalHub UI, AutoML UI, AutoRAG UI, kube-rbac-proxy).
No single node has room.

User is scaling up the ROSA nodepool via OCM console / ROSA CLI. Once new
capacity is online the pending pods should schedule themselves and the
DSC will reach `Ready=True` without further action.

To confirm post-scale, run:

```
oc get datascienceclusters.datasciencecluster.opendatahub.io default-dsc -o jsonpath='{.status.phase}'  # expect Ready
oc get pods -n redhat-ods-applications | grep -E '0/|Pending'  # expect empty
oc get route -n redhat-ods-applications rhods-dashboard  # confirm 200 via the host
```

## Recovery / backup location

`./backup-20260515T085706Z/`

- `config/` — DSCI raw, OperatorGroup raw, EA-1 CSV, Authentication, InstallPlans, namespaces, the stale Subscription we deleted, and SUBSCRIPTION_STATE.md noting that no Subscription existed at start.
- `workloads/` — Notebooks, InferenceServices, ServingRuntimes, DSPA, ModelRegistries, the `model-test/model1` connection Secret, all PVCs. `rag-blueprint` entries here reference an already-terminating namespace and won't reapply usefully.
- `reapply/` — cleaned manifests in apply order: `00-operatorgroup.yaml`, `02-dsc-default-managed.yaml` (v2!), `03-subscription-rhods-operator.yaml`, `10-notebooks.yaml`, `11-servingruntimes.yaml`, `12-inferenceservices.yaml`, `13-dspa.yaml`, `14-modelregistries.yaml`. We did NOT need to reapply 10–14 — soft reinstall preserved the CR instances.

## Gotchas for the next session

1. **Don't trust `oc get subscription` without FQN.** The bare name resolves to
   `subscriptions.messaging.knative.dev` (Knative) instead of
   `subscriptions.operators.coreos.com` (OLM) on this cluster. Always use
   `oc get subscriptions.operators.coreos.com -n redhat-ods-operator` or the
   `subs` short name.
2. **The 3.3-era `delete-self-managed-odh` ConfigMap trigger is dead in 3.4.**
   The opendatahub-operator no longer watches for it. Use manual uninstall:
   delete CSV → wait for pods → reapply Subscription. The CSV deletion is
   safe because the RHOAI CRDs do NOT have ownerReferences back to the CSV,
   so CR instances survive. OLM does GC the CSV-owned webhooks automatically.
3. **Use the v2 DSC API**, not v1. v1's validator rejects `managementState: Managed`
   with the misleading message "Managed is no longer supported as a managementState".
   v2 accepts it for every component except `kueue`, which uses `Unmanaged`.
4. **Cluster auth on this kubeconfig.** The `cluster-admin/...` user has a literal
   placeholder token "REDACTED" (8 chars) — useless. The `sp/...` user has
   effective cluster-admin (`oc auth can-i '*' '*'` → yes) and is what we used
   throughout. The handover originally said `sp` lacked privileges — stale.

## Done in this session

- Verified egress + installed `oc` 4.21.15 (arm64)
- Logged in as `sp` (cluster-admin equivalent), surveyed cluster
- Backed up cluster objects + workload CRs
- Deleted EA-1 CSV + cleaned trigger CM + cleaned stale Subscription/InstallPlan
- Recreated Subscription on `stable-3.4`, OLM installed `rhods-operator.3.4.0`
- Applied DSC v2 with the right managementState values
- Verified `model-test` and `rhoai-model-registries` workloads survived
- Confirmed `rag-blueprint` deletion was pre-existing, not caused by upgrade
- Identified cluster capacity as the only remaining blocker

## Not done (handed back to user)

- Scale up ROSA nodepool to free CPU for `rhods-dashboard` × 2 (~3.6 CPU each),
  notebook controllers (~CPU-light), and `spark-operator-controller`.
