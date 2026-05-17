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
    "Argumentuj VÝHRADNĚ údaji z dodaných důvodových kódů a hodnot ve faktuře. "
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
                           reason_codes: Iterable[str]) -> str:
    """Render one invoice row + its deterministic reasons as a compact CZ
    bullet block. Keeping it short = cheaper tokens + less room to drift."""
    rc = list(reason_codes)
    rc_block = "\n".join(f"  - {c}" for c in rc) if rc else "  - (žádné)"
    # Only print fields a controller would actually look at — avoids the model
    # hallucinating around irrelevant columns.
    cared = ("invoice_id", "vendor", "amount_czk", "category", "issued_on",
             "due_on", "po_number", "approver")
    fields = "\n".join(
        f"  - {k}: {row[k]}" for k in cared if k in row and row[k] not in (None, "")
    )
    return (
        f"Faktura:\n{fields}\n"
        f"Anomaly score: {anomaly_score:.3f}\n"
        f"Důvodové kódy (deterministické):\n{rc_block}"
    )


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
    max_tokens: int = 220,
    temperature: float = 0.2,
) -> str:
    """Return a 2-3 sentence Czech explanation for one flagged invoice.

    Defensive: any backend error returns a short fallback string instead of
    raising — the demo notebook should keep moving even if the LLM is down.
    """
    client = client or LLMClient()
    user = _format_row_for_prompt(row, anomaly_score, reason_codes)
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
    reason_col: str = "reason",
    score_col: str = "anomaly_score",
    progress: bool = True,
):
    """Add a 'vysvětlení' column for the top-N rows of a review queue.

    Returns a list of strings of length `top_n`. Caller assigns it to
    `queue.loc[:top_n-1, 'vysvětlení']`. Linear (one POST per row) — for
    top_n=25 against Phi-4 on A10G this is ~1-2 minutes; well within a
    workshop demo window.
    """
    client = client or LLMClient()
    n = min(top_n, len(queue_df))
    out: list[str] = []
    t0 = time.time()
    for i in range(n):
        row = queue_df.iloc[i].to_dict()
        text = llm_explain_row(
            row,
            anomaly_score=float(row.get(score_col, 0.0) or 0.0),
            reason_codes=_split_reason_codes(row.get(reason_col, "")),
            client=client,
        )
        out.append(text)
        if progress:
            elapsed = time.time() - t0
            print(f"  [{i + 1:2}/{n}] {elapsed:5.1f}s  "
                  f"{row.get('vendor', '?'):<24}  →  {text[:70]}"
                  f"{'…' if len(text) > 70 else ''}")
    return out


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
