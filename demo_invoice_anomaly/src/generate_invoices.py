"""Generate a synthetic invoice dataset for the workshop demo.

The dataset simulates 12 months of accounts-payable invoices for a mid-sized
company. Most rows are 'normal'; a small fraction are anomalies that a
controller would want to flag. The anomaly types deliberately span several
categories so the demo can show that anomaly detection picks up patterns
that simple rules would miss.

Anomaly types injected (~3% of rows total):
  - duplicate:        same vendor + amount within 7 days
  - round_sum:        suspiciously round invoice (e.g. 50 000 CZK)
  - off_hours:        booked on weekend or holiday
  - new_vendor_high:  brand-new vendor, unusually high first invoice
  - amount_outlier:   amount far above vendor's typical range
  - missing_po:       no purchase order reference for high-value invoice
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

VENDORS = [
    # (name, category, typical_amount_mean, typical_amount_std)
    ("ČEZ Distribuce a.s.",          "utilities",    45_000,  8_000),
    ("Pražská plynárenská a.s.",      "utilities",    18_000,  4_000),
    ("O2 Czech Republic",            "telecom",      12_000,  2_500),
    ("T-Mobile Czech Republic",      "telecom",       9_500,  2_000),
    ("Alza.cz a.s.",                 "office_supply", 6_500,  3_000),
    ("Office Depot s.r.o.",          "office_supply", 4_200,  1_500),
    ("Makro Cash & Carry",           "catering",      8_700,  3_500),
    ("Lagardère Travel Retail",      "travel",       11_500,  6_000),
    ("ČSA a.s.",                     "travel",       18_400,  9_000),
    ("Booking.com",                  "travel",        7_800,  4_500),
    ("Deloitte Advisory s.r.o.",     "consulting",   78_000, 22_000),
    ("PwC Česká republika",          "consulting",   84_000, 24_000),
    ("KPMG Česká republika",         "consulting",   72_000, 19_000),
    ("Kinstellar s.r.o.",            "legal",        56_000, 14_000),
    ("Havel & Partners",             "legal",        61_000, 18_000),
    ("AWS EMEA SARL",                "cloud",        42_000, 12_000),
    ("Microsoft Ireland Operations", "cloud",        38_000, 10_000),
    ("Google Cloud EMEA",            "cloud",        35_000,  9_500),
    ("Red Hat Czech s.r.o.",         "software",     95_000, 15_000),
    ("JetBrains s.r.o.",             "software",      8_400,  2_200),
    ("DHL Express",                  "logistics",    14_500,  5_500),
    ("PPL CZ s.r.o.",                "logistics",     9_800,  3_800),
    ("Manpower Group s.r.o.",        "hr_services", 110_000, 28_000),
    ("Grafton Recruitment",          "hr_services",  95_000, 24_000),
    ("Engie Services a.s.",          "facility",     28_000,  7_000),
]

CATEGORIES = sorted({v[1] for v in VENDORS})

# Czech public holidays 2025 (subset that matters for weekday checks)
HOLIDAYS = {
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 21),
    date(2025, 5, 1), date(2025, 5, 8), date(2025, 7, 5),
    date(2025, 7, 6), date(2025, 9, 28), date(2025, 10, 28),
    date(2025, 11, 17), date(2025, 12, 24), date(2025, 12, 25),
    date(2025, 12, 26),
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _random_business_date(start: date, end: date) -> date:
    """Pick a uniformly random business day in [start, end]."""
    span = (end - start).days
    while True:
        d = start + timedelta(days=int(RNG.integers(0, span + 1)))
        if d.weekday() < 5 and d not in HOLIDAYS:
            return d


def _round_amount(value: float) -> float:
    """Round to two decimals like a real invoice would."""
    return float(np.round(value, 2))


def _normal_invoice(invoice_id: int, start: date, end: date) -> dict:
    vendor, category, mean, std = random.choice(VENDORS)
    amount = max(500.0, RNG.normal(mean, std))
    issued = _random_business_date(start, end)
    due = issued + timedelta(days=int(RNG.integers(14, 45)))
    return {
        "invoice_id":   f"INV-{invoice_id:06d}",
        "vendor":       vendor,
        "category":     category,
        "amount_czk":   _round_amount(amount),
        "issued_on":    issued,
        "due_on":       due,
        "po_number":    f"PO-{RNG.integers(10000, 99999)}" if RNG.random() > 0.05 else "",
        "cost_center":  f"CC-{RNG.integers(100, 400):03d}",
        "approver":     random.choice(["jnovak", "kvarga", "msrnec", "tklimes", "ldvorak"]),
        "is_anomaly":   0,
        "anomaly_type": "",
    }


def _inject_anomaly(row: dict, kind: str, start: date, end: date) -> dict:
    row = dict(row)
    row["is_anomaly"] = 1
    row["anomaly_type"] = kind

    if kind == "round_sum":
        row["amount_czk"] = float(random.choice([50_000, 100_000, 150_000, 200_000, 250_000]))
    elif kind == "off_hours":
        # Force weekend or holiday issue date
        d = _random_business_date(start, end)
        # Roll to nearest weekend day
        delta = (5 - d.weekday()) if d.weekday() < 5 else 0
        row["issued_on"] = d + timedelta(days=delta)
        row["due_on"] = row["issued_on"] + timedelta(days=int(RNG.integers(14, 45)))
    elif kind == "new_vendor_high":
        # Replace vendor with a one-off
        row["vendor"] = f"AdHoc Trading s.r.o. #{RNG.integers(1000, 9999)}"
        row["category"] = "consulting"
        row["amount_czk"] = _round_amount(RNG.uniform(180_000, 400_000))
    elif kind == "amount_outlier":
        row["amount_czk"] = _round_amount(row["amount_czk"] * RNG.uniform(6, 12))
    elif kind == "missing_po":
        row["amount_czk"] = max(row["amount_czk"], _round_amount(RNG.uniform(120_000, 300_000)))
        row["po_number"] = ""
    # duplicate handled at dataset level (needs a sibling row)
    return row


def generate(n_rows: int, anomaly_rate: float, start: date, end: date) -> pd.DataFrame:
    """Produce a dataframe of n_rows invoices."""
    n_anomaly = int(n_rows * anomaly_rate)
    n_normal = n_rows - n_anomaly

    rows: list[dict] = []
    next_id = 1
    for _ in range(n_normal):
        rows.append(_normal_invoice(next_id, start, end))
        next_id += 1

    anomaly_kinds = ["round_sum", "off_hours", "new_vendor_high",
                     "amount_outlier", "missing_po", "duplicate"]
    for _ in range(n_anomaly):
        base = _normal_invoice(next_id, start, end)
        next_id += 1
        kind = random.choice(anomaly_kinds)
        if kind == "duplicate":
            # add the original
            rows.append(base)
            dup = dict(base)
            dup["invoice_id"] = f"INV-{next_id:06d}"
            next_id += 1
            dup["issued_on"] = base["issued_on"] + timedelta(days=int(RNG.integers(1, 6)))
            dup["due_on"]    = dup["issued_on"] + timedelta(days=int(RNG.integers(14, 45)))
            dup["is_anomaly"] = 1
            dup["anomaly_type"] = "duplicate"
            rows.append(dup)
        else:
            rows.append(_inject_anomaly(base, kind, start, end))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # Re-issue stable invoice IDs in order
    df["invoice_id"] = [f"INV-{i + 1:06d}" for i in range(len(df))]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent.parent / "data" / "invoices.csv",
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end",   default="2025-12-31")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    df = generate(args.rows, args.anomaly_rate, start, end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}")
    print(f"  anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.1%})")
    print("  by type:")
    print(df[df["is_anomaly"] == 1]["anomaly_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
