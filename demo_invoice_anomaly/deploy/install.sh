#!/usr/bin/env bash
# Bootstrap (or remove) the invoice-anomaly demo on the sa-ai cluster.
# Required:  `oc` on PATH and a kubeconfig pointed at the cluster.
# Optional:  `mc` (MinIO client). If absent, an in-cluster mc pod is used.
#
# Usage:
#   ./install.sh             # install (default)
#   ./install.sh install     # install (explicit)
#   ./install.sh uninstall   # remove demo workloads, keep namespace + PVCs
#   ./install.sh purge       # full wipe — also drops namespace + PVCs (DATA LOSS)
#   ./install.sh -h          # show this help

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NS="rh-ai-workshop"
ACTION="${1:-install}"
MC_IMAGE="quay.io/minio/mc:RELEASE.2024-11-21T17-21-54Z"

usage() {
  sed -n '2,11p' "$0"
  exit 0
}

# ---------------------------------------------------------------------------
# upload_csvs — push CSVs to MinIO without needing local 'mc'
#
# Two paths:
#   1) If local `mc` is on PATH, port-forward and use it (fast path).
#   2) Otherwise spin up an mc pod in-cluster, oc-cp the CSVs in, and
#      run `mc` from there. Requires only `oc` on the laptop.
# ---------------------------------------------------------------------------
upload_csvs() {
  local bucket="rhoai-workshop-invoices"
  local pw
  pw="$(oc -n "$NS" get secret minio-root -o jsonpath='{.data.rootPassword}' | base64 -d)"
  local user
  user="$(oc -n "$NS" get secret minio-root -o jsonpath='{.data.rootUser}' | base64 -d)"

  if command -v mc >/dev/null 2>&1; then
    echo "    using local 'mc' (port-forwarding 9000)"
    oc -n "$NS" port-forward svc/minio 9000:9000 >/dev/null 2>&1 &
    local pf=$!
    trap "kill $pf 2>/dev/null || true" RETURN
    sleep 3
    mc alias set wb "http://localhost:9000" "$user" "$pw"
    mc mb --ignore-existing "wb/$bucket"
    mc cp "$HERE/../data/invoices.csv"           "wb/$bucket/"
    mc cp "$HERE/../data/invoices_new_batch.csv" "wb/$bucket/"
    return
  fi

  echo "    local 'mc' not found — using in-cluster mc pod (only 'oc' needed)"
  oc -n "$NS" delete pod mc-upload --ignore-not-found --wait=true >/dev/null 2>&1 || true
  oc -n "$NS" run mc-upload \
    --image="$MC_IMAGE" \
    --restart=Never \
    --command -- sleep 600 >/dev/null

  echo "    waiting for mc-upload pod"
  oc -n "$NS" wait --for=condition=Ready pod/mc-upload --timeout=120s >/dev/null

  echo "    copying CSVs into pod"
  oc -n "$NS" cp "$HERE/../data/invoices.csv"           "mc-upload:/tmp/invoices.csv"
  oc -n "$NS" cp "$HERE/../data/invoices_new_batch.csv" "mc-upload:/tmp/invoices_new_batch.csv"

  echo "    running mc inside the cluster"
  oc -n "$NS" exec mc-upload -- sh -c "
    set -e
    mc alias set wb http://minio:9000 '$user' '$pw'
    mc mb --ignore-existing wb/$bucket
    mc cp /tmp/invoices.csv           wb/$bucket/
    mc cp /tmp/invoices_new_batch.csv wb/$bucket/
    mc ls wb/$bucket/
  "

  oc -n "$NS" delete pod mc-upload --wait=false >/dev/null
  echo "    uploaded CSVs"
}

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
do_install() {
  echo "==> Applying manifests"
  oc apply -f "$HERE/01-namespace.yaml"
  oc apply -f "$HERE/02-minio.yaml"
  oc apply -f "$HERE/03-data-connection.yaml"
  oc apply -f "$HERE/05-dspa.yaml"
  oc apply -f "$HERE/04-workbench.yaml"
  oc apply -f "$HERE/07-model-registry-rbac.yaml"

  echo "==> Waiting for MinIO"
  oc -n "$NS" rollout status deploy/minio --timeout=180s

  echo "==> Bootstrap bucket"
  oc apply -f "$HERE/06-bootstrap-data.yaml"
  oc -n "$NS" wait --for=condition=complete --timeout=120s job/bootstrap-bucket || true

  echo "==> Upload CSVs to MinIO"
  upload_csvs

  echo "==> Done."
  echo
  echo "Workbench URL:"
  oc -n "$NS" get route -l notebook-name=invoice-anomaly-wb -o jsonpath='{.items[0].spec.host}' 2>/dev/null \
    || echo "  (route still pending — check 'oc get route -n $NS' in a minute)"
  echo
  echo "MinIO console:"
  oc -n "$NS" get route minio-console -o jsonpath='{.spec.host}{"\n"}' 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# uninstall — remove demo workloads, keep namespace + PVCs (data preserved)
# ---------------------------------------------------------------------------
do_uninstall() {
  if ! oc get namespace "$NS" >/dev/null 2>&1; then
    echo "Namespace '$NS' does not exist — nothing to uninstall."
    return 0
  fi

  echo "==> Uninstalling demo workloads from namespace '$NS'"
  echo "    (PVCs and the namespace itself are preserved — use 'purge' to wipe)"

  # Reverse order of install, ignoring 'not found' so reruns are safe
  oc delete -f "$HERE/06-bootstrap-data.yaml" --ignore-not-found --wait=false
  oc delete -f "$HERE/07-model-registry-rbac.yaml" --ignore-not-found --wait=false
  oc delete -f "$HERE/04-workbench.yaml"      --ignore-not-found --wait=false
  oc delete -f "$HERE/05-dspa.yaml"           --ignore-not-found --wait=false
  oc delete -f "$HERE/03-data-connection.yaml" --ignore-not-found --wait=false
  oc delete -f "$HERE/02-minio.yaml"          --ignore-not-found --wait=false

  # Best-effort cleanup of pipeline runs / pods (DS Pipelines leftovers).
  # These are created by the DSPA controller, not by our manifests.
  echo "==> Cleaning up pipeline runs and finished pods"
  oc -n "$NS" delete pipelineruns --all       --ignore-not-found --wait=false 2>/dev/null || true
  oc -n "$NS" delete workflows.argoproj.io --all --ignore-not-found --wait=false 2>/dev/null || true
  oc -n "$NS" delete pods --field-selector=status.phase=Succeeded --ignore-not-found 2>/dev/null || true
  oc -n "$NS" delete pods --field-selector=status.phase=Failed    --ignore-not-found 2>/dev/null || true

  echo "==> Uninstall complete."
  echo
  echo "Remaining objects in namespace:"
  oc -n "$NS" get all,pvc,secret -o name 2>/dev/null | grep -v 'serviceaccount/builder\|serviceaccount/default\|serviceaccount/deployer\|secret/builder\|secret/default\|secret/deployer' || true
  echo
  echo "To also drop PVCs and the namespace itself, run: $0 purge"
}

# ---------------------------------------------------------------------------
# purge — full wipe (drops namespace, which removes PVCs and all data)
# ---------------------------------------------------------------------------
do_purge() {
  if ! oc get namespace "$NS" >/dev/null 2>&1; then
    echo "Namespace '$NS' does not exist — nothing to purge."
    return 0
  fi

  echo "*** WARNING ***"
  echo "This will DELETE namespace '$NS' including:"
  echo "  - all workloads (workbench, MinIO, DSPA, MariaDB, ...)"
  echo "  - all PVCs (uploaded invoices, MLflow runs, pipeline DB)"
  echo "  - all secrets, routes, configmaps in the namespace"
  echo
  read -rp "Type the namespace name ('$NS') to confirm: " CONFIRM
  if [[ "$CONFIRM" != "$NS" ]]; then
    echo "Aborted."
    return 1
  fi

  echo "==> Deleting namespace '$NS' (this can take 30–90s)"
  oc delete namespace "$NS" --wait=true
  echo "==> Purge complete."
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
case "$ACTION" in
  install)            do_install ;;
  uninstall|remove)   do_uninstall ;;
  purge|nuke)         do_purge ;;
  -h|--help|help)     usage ;;
  *) echo "Unknown action: $ACTION"; echo; usage ;;
esac
