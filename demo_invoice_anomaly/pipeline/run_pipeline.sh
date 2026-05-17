#!/usr/bin/env bash
#
# One-command trigger for the invoice-anomaly Data Science Pipeline.
#
# Workshop usage:
#   ./run_pipeline.sh
#
# What it does (in order):
#   1) Compiles the pipeline yaml from the .py source if either is missing
#      or .py is newer than .yaml.
#   2) Finds or creates the `invoice-anomaly` pipeline on the cluster.
#   3) Uploads the current yaml as a new pipeline version named
#      `<short-git-sha>-<UTC-timestamp>` (or `local-<timestamp>` if not in git).
#   4) Creates a run referencing that fresh version, in the Default experiment.
#   5) Prints the Dashboard URL so the audience can watch the DAG live.
#
# Auth: uses `oc whoami -t` if available locally, otherwise reads the
# in-cluster SA token at /var/run/secrets/kubernetes.io/serviceaccount/token
# (so the same script works from the workbench's terminal).
#
# Requires `oc`, `curl`, `python3`. No kfp SDK needed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — adjust if you re-deploy under a different namespace/registry.
# ---------------------------------------------------------------------------
NAMESPACE="${NAMESPACE:-rh-ai-workshop}"
PIPELINE_NAME="${PIPELINE_NAME:-invoice-anomaly}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Default}"
PIPELINE_API_HOST="${PIPELINE_API_HOST:-}"   # auto-discover via `oc get route` if empty
CLUSTER_HOST="${CLUSTER_HOST:-}"             # for the Dashboard URL banner (auto-discovered)

# Where to find the pipeline yaml relative to this script.
HERE="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_PY="$HERE/invoice_anomaly_pipeline.py"
PIPELINE_YAML="$HERE/invoice_anomaly_pipeline.yaml"

# ---------------------------------------------------------------------------
# 1) Compile if needed.
# ---------------------------------------------------------------------------
need_compile=0
if [[ ! -f "$PIPELINE_YAML" ]]; then
  need_compile=1
elif [[ "$PIPELINE_PY" -nt "$PIPELINE_YAML" ]]; then
  need_compile=1
fi
if (( need_compile )); then
  echo "==> Compiling $PIPELINE_PY -> invoice_anomaly_pipeline.yaml"
  (cd "$HERE" && python3 "$PIPELINE_PY")
fi

# ---------------------------------------------------------------------------
# 2) Detect environment: laptop (oc-logged-in) vs in-cluster (workbench pod).
# ---------------------------------------------------------------------------
IN_CLUSTER=0
if [[ -r /var/run/secrets/kubernetes.io/serviceaccount/token ]] && \
   [[ -r /var/run/secrets/kubernetes.io/serviceaccount/namespace ]]; then
  IN_CLUSTER=1
fi

# Auth: prefer the in-cluster SA token when we're in a pod (the workbench's
# `oc whoami -t` also returns it, but reading the projected file is one less
# external process and works without `oc`).
if (( IN_CLUSTER )); then
  TOKEN="$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)"
  WHO="ServiceAccount $(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)/$(oc whoami 2>/dev/null | sed 's|.*:||' || echo unknown)"
  echo "==> Auth: in-cluster SA token ($WHO)"
elif command -v oc >/dev/null && oc whoami -t >/dev/null 2>&1; then
  TOKEN="$(oc whoami -t)"
  echo "==> Auth: oc user $(oc whoami)"
else
  echo "ERROR: no oc login and no in-cluster SA token — cannot authenticate." >&2
  exit 1
fi

# API host: env override > external route > in-cluster Service DNS.
# In-cluster fallback uses HTTPS:8888 with `-k` because the Service cert is
# self-signed; we stay inside the cluster trust boundary so this is fine.
# (Kept as a string, not an array — bash 3.2 on macOS errors on empty-array
# expansion under `set -u`.)
CURL_INSECURE=""
if [[ -z "$PIPELINE_API_HOST" ]]; then
  if command -v oc >/dev/null; then
    PIPELINE_API_HOST="$(oc -n "$NAMESPACE" get route ds-pipeline-dspa \
      -o jsonpath='{.spec.host}' 2>/dev/null || true)"
  fi
fi
if [[ -n "$PIPELINE_API_HOST" ]]; then
  ROUTE="https://$PIPELINE_API_HOST"
elif (( IN_CLUSTER )); then
  # Workbench SA usually can't read routes; the Service is reachable and
  # accepts the same SA token.
  ROUTE="https://ds-pipeline-dspa.$NAMESPACE.svc.cluster.local:8888"
  CURL_INSECURE="-k"
  echo "==> Note: workbench SA cannot read routes — using in-cluster Service DNS."
else
  echo "ERROR: could not auto-discover ds-pipeline-dspa route in $NAMESPACE." >&2
  echo "       Pass PIPELINE_API_HOST=<host> explicitly." >&2
  exit 1
fi
echo "==> API:  $ROUTE"

if [[ -z "$CLUSTER_HOST" ]] && command -v oc >/dev/null; then
  CLUSTER_HOST="$(oc -n openshift-console get route console -o jsonpath='{.spec.host}' 2>/dev/null || true)"
fi

# Helpers — every curl uses --max-time so a network hiccup does not hang the demo.
# $CURL_INSECURE intentionally unquoted: empty → no flag; "-k" → single flag.
curlh() { curl -sS --max-time 30 $CURL_INSECURE -H "Authorization: Bearer $TOKEN" "$@"; }

# ---------------------------------------------------------------------------
# 3) Find or create the pipeline.
# ---------------------------------------------------------------------------
echo "==> Looking up pipeline '$PIPELINE_NAME'"
PIPELINE_ID=$(curlh "$ROUTE/apis/v2beta1/pipelines?page_size=100" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for p in data.get('pipelines', []):
    if p.get('name') == '$PIPELINE_NAME':
        print(p['pipeline_id']); break
")

if [[ -z "$PIPELINE_ID" ]]; then
  echo "    not found — uploading new pipeline"
  RESP=$(curlh -F "uploadfile=@$PIPELINE_YAML" \
    "$ROUTE/apis/v2beta1/pipelines/upload?name=$PIPELINE_NAME")
  PIPELINE_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['pipeline_id'])")
  echo "    created: $PIPELINE_ID"
else
  echo "    found: $PIPELINE_ID"
fi

# ---------------------------------------------------------------------------
# 4) Upload a new version of the current yaml.
# ---------------------------------------------------------------------------
SHA=""
if command -v git >/dev/null && git -C "$HERE" rev-parse --short HEAD >/dev/null 2>&1; then
  SHA="$(git -C "$HERE" rev-parse --short HEAD)"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
VERSION_NAME="${SHA:-local}-$STAMP"

echo "==> Uploading pipeline version '$VERSION_NAME'"
RESP=$(curlh -F "uploadfile=@$PIPELINE_YAML" \
  "$ROUTE/apis/v2beta1/pipelines/upload_version?name=$VERSION_NAME&pipelineid=$PIPELINE_ID")
VERSION_ID=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['pipeline_version_id'])")
echo "    version_id: $VERSION_ID"

# ---------------------------------------------------------------------------
# 5) Look up experiment id.
# ---------------------------------------------------------------------------
EXPERIMENT_ID=$(curlh "$ROUTE/apis/v2beta1/experiments?page_size=100" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for e in data.get('experiments', []):
    if e.get('display_name') == '$EXPERIMENT_NAME':
        print(e['experiment_id']); break
")
if [[ -z "$EXPERIMENT_ID" ]]; then
  echo "ERROR: experiment '$EXPERIMENT_NAME' not found." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 6) Create the run.
# ---------------------------------------------------------------------------
RUN_NAME="invoice-anomaly-$STAMP"
HOST_TAG="${HOSTNAME:-$(cat /etc/hostname 2>/dev/null || echo unknown)}"
echo "==> Creating run '$RUN_NAME'"
cat > /tmp/_pipeline_run.json <<JSON
{
  "display_name": "$RUN_NAME",
  "description": "Triggered by run_pipeline.sh from $(whoami)@$HOST_TAG",
  "pipeline_version_reference": {
    "pipeline_id": "$PIPELINE_ID",
    "pipeline_version_id": "$VERSION_ID"
  },
  "experiment_id": "$EXPERIMENT_ID",
  "runtime_config": {"parameters": {}}
}
JSON
RUN_RESP=$(curlh -H "Content-Type: application/json" \
  -d @/tmp/_pipeline_run.json \
  "$ROUTE/apis/v2beta1/runs")
RUN_ID=$(echo "$RUN_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")
rm -f /tmp/_pipeline_run.json
echo "    run_id: $RUN_ID"

# ---------------------------------------------------------------------------
# 7) Print Dashboard URL.
# ---------------------------------------------------------------------------
if [[ -n "$CLUSTER_HOST" ]]; then
  DASH=$(echo "$CLUSTER_HOST" | sed 's/^console-openshift-console/rhods-dashboard-redhat-ods-applications/')
  echo
  echo "    ┌──────────────────────────────────────────────────────────────"
  echo "    │ Watch the run in RHOAI Dashboard:"
  echo "    │   https://$DASH/projects/$NAMESPACE?section=pipelines-runs"
  echo "    │ Or open the run directly:"
  echo "    │   https://$DASH/pipelineRuns/$NAMESPACE/pipelineRun/view/$RUN_ID"
  echo "    └──────────────────────────────────────────────────────────────"
fi
echo "done."
