"""RAG-lite context retrieval over vendor invoice history.

The LLM "vysvětlovač" (explain_with_llm) by default explains an anomaly
flag using only the row's own fields and the deterministic reason codes.
That's accurate but generic — it can't say "vendor X usually bills under
50k, this one is 17× over median" because it has no history.

This module retrieves a structured summary of a vendor's past invoices and
formats it as a compact Czech context block that the LLM can ground its
explanation in. Retrieval here is structural (`df[df.vendor == V]`), not
semantic — that's the right shape for tabular evidence. The pattern
generalises: replace the dataframe filter with a vector-DB query and you
have full RAG over unstructured documents.

Why training-set rows and not Feast historical features?
- Feast's FeatureView publishes the 9 numeric features the model uses.
  For a controller-readable explanation we want raw fields (vendor name,
  amount, category, dates), which aren't in the FV schema.
- In production this corpus would live in a warehouse / lake / vector DB
  alongside (not inside) Feast. Demo uses `invoices.csv` as the stand-in.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Loading the corpus
# ---------------------------------------------------------------------------


def load_historical_invoices(
    s3_client=None,
    bucket: Optional[str] = None,
    local_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load the invoice history corpus used for RAG retrieval.

    Same shape as the training set (output of `invoice_features.load_invoices`).
    Either from MinIO (cluster path) or from local disk (laptop path).
    """
    # Avoid a hard dependency loop — invoice_features imports nothing from
    # us, so this late import is safe and saves an extra arg to the caller.
    from invoice_features import load_invoices

    if s3_client is not None and bucket:
        return load_invoices("invoices.csv", s3_client=s3_client, bucket=bucket)
    return load_invoices(local_path or "../data/invoices.csv")


# ---------------------------------------------------------------------------
# Per-vendor retrieval + summary
# ---------------------------------------------------------------------------


def _ratio_str(n: int, total: int) -> str:
    return f"{n}/{total}"


def _cz_int(value: float) -> str:
    """Format a number as a Czech-style 'thousands grouped' integer."""
    return f"{int(round(value)):,}".replace(",", " ")


def vendor_history(
    vendor: str,
    history_df: pd.DataFrame,
    *,
    current_amount: Optional[float] = None,
    exclude_invoice_id: Optional[str] = None,
) -> dict:
    """Summarise a vendor's invoice history into a small JSON-ish dict.

    The dict is consumed by `format_history_for_prompt` to produce a Czech
    text block for the LLM. Keeping a structured intermediate makes it
    easy to test, to swap the prompt format, or to surface the same
    summary in the review queue CSV.
    """
    sub = history_df[history_df["vendor"] == vendor]
    if exclude_invoice_id is not None:
        sub = sub[sub["invoice_id"] != exclude_invoice_id]
    n = len(sub)

    if n == 0:
        return {
            "vendor": vendor,
            "count": 0,
            "is_new_vendor": True,
        }

    amounts = sub["amount_czk"].astype(float)
    summary = {
        "vendor": vendor,
        "count": int(n),
        "is_new_vendor": False,
        "amount_median": float(amounts.median()),
        "amount_p10": float(amounts.quantile(0.10)),
        "amount_p90": float(amounts.quantile(0.90)),
        "amount_max": float(amounts.max()),
        "typical_category": sub["category"].mode().iat[0],
        "typical_category_count": int((sub["category"] == sub["category"].mode().iat[0]).sum()),
        "missing_po_count": int(sub["po_number"].fillna("").astype(str).eq("").sum()),
        "weekend_count": int(pd.to_datetime(sub["issued_on"]).dt.weekday.ge(5).sum()),
        "round_sum_count": int(((amounts % 10_000 == 0) & (amounts >= 10_000)).sum()),
        "last_seen": str(pd.to_datetime(sub["issued_on"]).max().date()),
    }

    if current_amount is not None and summary["amount_median"] > 0:
        summary["multiple_of_median"] = float(current_amount / summary["amount_median"])

    return summary


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_history_for_prompt(summary: dict) -> str:
    """Render the dict from `vendor_history` as a compact Czech text block.

    Empty / new-vendor case is intentionally short — the signal there is
    *the absence* of history.
    """
    if summary.get("is_new_vendor", False):
        return (
            f"Historie dodavatele {summary['vendor']}: žádná předchozí "
            f"faktura, dodavatel je nový."
        )

    n = summary["count"]
    lines = [f"Historie dodavatele {summary['vendor']} (z {n} předchozích faktur):"]
    lines.append(
        f"- typická částka: {_cz_int(summary['amount_median'])} Kč "
        f"(rozpětí {_cz_int(summary['amount_p10'])}–{_cz_int(summary['amount_p90'])} Kč, "
        f"max {_cz_int(summary['amount_max'])} Kč)"
    )
    lines.append(
        f"- typická kategorie: {summary['typical_category']} "
        f"({_ratio_str(summary['typical_category_count'], n)})"
    )
    lines.append(f"- bez PO: {_ratio_str(summary['missing_po_count'], n)}")
    lines.append(f"- o víkendu: {_ratio_str(summary['weekend_count'], n)}")
    lines.append(f"- kulaté částky (≥ 10 000): {_ratio_str(summary['round_sum_count'], n)}")
    lines.append(f"- poslední faktura: {summary['last_seen']}")

    if "multiple_of_median" in summary:
        m = summary["multiple_of_median"]
        if m >= 2 or m <= 0.5:
            lines.append(
                f"- aktuální faktura je **{m:.1f}× nad/pod mediánem** jeho historie"
            )
    return "\n".join(lines)


def vendor_context_for_row(
    row: dict,
    history_df: pd.DataFrame,
) -> str:
    """Convenience: produce the formatted Czech context block straight from
    a (review-queue) row dict + corpus."""
    summary = vendor_history(
        row.get("vendor", ""),
        history_df,
        current_amount=float(row.get("amount_czk", 0.0) or 0.0),
        exclude_invoice_id=row.get("invoice_id"),
    )
    return format_history_for_prompt(summary)
