"""LLM 'vysvětlovač' for the invoice review queue.

For each row a controller would have to read, the LLM produces a 2-3 sentence
Czech explanation of *why* the row was flagged — grounded strictly in the
row's own data and the deterministic reason codes already produced by
`invoice_features.explain_row`.

Backend: any OpenAI-compatible chat-completions endpoint. We default to the
in-cluster vLLM predictor of `phi-4-quantizedw8a8-version-1` in the
`model-test` namespace on the sa-ai workshop cluster. Override via
environment:

    LLM_ENDPOINT   base URL ending in /v1 (default: phi-4 predictor :8080)
    LLM_MODEL      OpenAI-style model id (default: phi-4-quantizedw8a8-version-1)
    LLM_API_KEY    optional bearer for the API (vLLM ignores it by default)

Pure stdlib (urllib + json) — no extra pip install in the workbench.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Defaults — match the live state on sa-ai (model-test/phi-4-...)
# ---------------------------------------------------------------------------

# NB: The IS advertises `http://<name>-predictor...svc.cluster.local` (port 80)
# but that is a headless KServe service with no proxy on :80 — clients must
# hit the pod's listening port 8080 directly. This default reflects that.
DEFAULT_ENDPOINT = (
    "http://phi-4-quantizedw8a8-version-1-predictor.model-test"
    ".svc.cluster.local:8080/v1"
)
DEFAULT_MODEL = "phi-4-quantizedw8a8-version-1"

SYSTEM_PROMPT = (
    "Jsi auditor faktur. Tvým úkolem je controllerovi stručně vysvětlit, "
    "proč právě tato faktura vyšla jako podezřelá. "
    "Odpověz MAXIMÁLNĚ třemi krátkými větami v jednom odstavci — bez odrážek "
    "a bez výčtu za prvé / za druhé. "
    "Argumentuj VÝHRADNĚ údaji z dodaných důvodových kódů, hodnot ve faktuře, "
    "a (pokud je k dispozici) historie dodavatele. "
    "Pokud máš k dispozici historii, OPŘI vysvětlení o konkrétní čísla z ní — "
    "například dodavatel X obvykle účtuje 35 000 Kč, dnes 1 240 000 Kč. "
    "Nevymýšlej si nic, co ve vstupu není. Žádné rady, žádné doporučení akcí — "
    "jen popis důvodu podezření. Anomaly score je v rozsahu 0..1, kde vyšší = "
    "vyšší podezření z anomálie."
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class LLMClient:
    """Minimal OpenAI-compatible chat-completions client (stdlib only)."""

    endpoint: str = ""
    model: str = ""
    api_key: Optional[str] = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.endpoint = (self.endpoint or os.environ.get("LLM_ENDPOINT")
                         or DEFAULT_ENDPOINT).rstrip("/")
        self.model = self.model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        if self.api_key is None:
            self.api_key = os.environ.get("LLM_API_KEY") or None

    # -- low-level -------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def list_models(self) -> dict:
        """GET /models — useful as a connectivity smoke-test."""
        req = urllib.request.Request(
            f"{self.endpoint}/models", headers=self._headers(), method="GET"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 200,
        top_p: float = 0.9,
    ) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }).encode()
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=body, headers=self._headers(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _format_row_for_prompt(row: dict, anomaly_score: float,
                           reason_codes: Iterable[str],
                           context_block: Optional[str] = None) -> str:
    """Render one invoice row + its deterministic reasons as a compact CZ
    bullet block. Keeping it short = cheaper tokens + less room to drift.

    `context_block`, if given, is RAG-style retrieved context (typically a
    vendor-history summary from `rag_context.format_history_for_prompt`).
    It's appended below the invoice fields so the model knows it's
    additional evidence, not part of the current row.
    """
    rc = list(reason_codes)
    rc_block = "\n".join(f"  - {c}" for c in rc) if rc else "  - (žádné)"
    # Only print fields a controller would actually look at — avoids the model
    # hallucinating around irrelevant columns.
    cared = ("invoice_id", "vendor", "amount_czk", "category", "issued_on",
             "due_on", "po_number", "approver")
    fields = "\n".join(
        f"  - {k}: {row[k]}" for k in cared if k in row and row[k] not in (None, "")
    )
    parts = [
        f"Faktura:\n{fields}",
        f"Anomaly score: {anomaly_score:.3f}",
        f"Důvodové kódy (deterministické):\n{rc_block}",
    ]
    if context_block:
        parts.append(context_block)
    return "\n".join(parts)


def _split_reason_codes(reason: str) -> list[str]:
    """invoice_features.explain_row joins reasons with '; ' — undo that so the
    LLM gets a clean bullet list."""
    if not reason:
        return []
    return [r.strip() for r in str(reason).split(";") if r.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def llm_explain_row(
    row: dict,
    anomaly_score: float,
    reason_codes: Iterable[str],
    *,
    client: Optional[LLMClient] = None,
    context_block: Optional[str] = None,
    max_tokens: int = 220,
    temperature: float = 0.2,
) -> str:
    """Return a 2-3 sentence Czech explanation for one flagged invoice.

    `context_block` is optional RAG-style retrieved context. Typically a
    vendor history summary from `rag_context.vendor_context_for_row`.
    When given, the LLM grounds its explanation in those concrete numbers.

    Defensive: any backend error returns a short fallback string instead of
    raising — the demo notebook should keep moving even if the LLM is down.
    """
    client = client or LLMClient()
    user = _format_row_for_prompt(row, anomaly_score, reason_codes, context_block)
    try:
        resp = client.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        return f"(LLM nedostupné: {e.reason})"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"(LLM odpověď v neočekávaném tvaru: {e})"
    except Exception as e:  # noqa: BLE001
        return f"(LLM selhalo: {type(e).__name__}: {e})"


def explain_queue(
    queue_df,
    *,
    top_n: int = 25,
    client: Optional[LLMClient] = None,
    history_df=None,
    reason_col: str = "reason",
    score_col: str = "anomaly_score",
    max_workers: int = 5,
    progress: bool = True,
):
    """Add a 'vysvětlení' column for the top-N rows of a review queue.

    If `history_df` is given (the vendor history corpus, typically the
    training-set invoices), we compute a RAG-style vendor summary per row
    and pass it to the LLM as additional evidence. Without it the LLM
    explains from the row + reason codes alone.

    `max_workers` controls in-flight concurrency against the LLM endpoint.
    vLLM 0.18 happily serves multiple chat-completion requests in parallel
    sharing the same KV-cache, so for top_n=25 the wall-clock time goes
    from ~3-4 min (sequential) down to ~30-45 s on Phi-4 (or ~10-15 s on
    Apertus-8B). Set max_workers=1 to disable concurrency for debugging.

    Returns a list of strings of length `top_n`, **in queue order** —
    caller assigns it to `queue.loc[:top_n-1, 'vysvětlení']` regardless of
    completion order.
    """
    client = client or LLMClient()
    # Late import so the demo runs even without the history corpus loaded.
    if history_df is not None:
        from rag_context import vendor_context_for_row  # noqa: WPS433

    n = min(top_n, len(queue_df))
    rows = [queue_df.iloc[i].to_dict() for i in range(n)]

    def _one(idx_row: tuple[int, dict]) -> tuple[int, str]:
        idx, row = idx_row
        ctx = vendor_context_for_row(row, history_df) if history_df is not None else None
        text = llm_explain_row(
            row,
            anomaly_score=float(row.get(score_col, 0.0) or 0.0),
            reason_codes=_split_reason_codes(row.get(reason_col, "")),
            context_block=ctx,
            client=client,
        )
        return idx, text

    out: list[Optional[str]] = [None] * n
    t0 = time.time()
    completed = 0
    workers = max(1, min(max_workers, n))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, (i, rows[i])) for i in range(n)]
        # Print rows as they FINISH (not as they're submitted) — completion
        # order can differ from queue order, but the final list is in order.
        for fut in concurrent.futures.as_completed(futures):
            idx, text = fut.result()
            out[idx] = text
            completed += 1
            if progress:
                elapsed = time.time() - t0
                vendor = rows[idx].get("vendor", "?")
                print(f"  [{completed:2}/{n}] {elapsed:5.1f}s  "
                      f"row={idx:>2}  {vendor:<24}  →  {text[:60]}"
                      f"{'…' if len(text) > 60 else ''}")

    return [t or "(prázdná odpověď)" for t in out]


# ---------------------------------------------------------------------------
# Smoke test — `python -m src.explain_with_llm`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    client = LLMClient()
    print(f"endpoint = {client.endpoint}")
    print(f"model    = {client.model}")
    try:
        models = client.list_models()
        ids = [m.get("id") for m in models.get("data", [])]
        print(f"reachable, models = {ids}")
    except Exception as e:  # noqa: BLE001
        print(f"GET /models failed: {e}")
        sys.exit(2)

    sample_row = {
        "invoice_id": "INV-2025-08842",
        "vendor": "NOVA-LTD",
        "amount_czk": 98_765,
        "category": "marketing",
        "issued_on": "2025-10-04",
        "due_on": "2025-11-04",
        "po_number": None,
        "approver": "auto",
    }
    txt = llm_explain_row(
        sample_row,
        anomaly_score=0.92,
        reason_codes=["podezřele kulatá částka",
                      "zaúčtováno o víkendu / svátku",
                      "nový dodavatel s vysokou částkou"],
        client=client,
    )
    print("\n=== Smoke explanation ===\n" + txt)
