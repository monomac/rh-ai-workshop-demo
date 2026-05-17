"""Feast entities for the invoice-anomaly demo.

One entity: the invoice itself, keyed on its document number. Every feature
in feature_views.py is keyed by `invoice_id`.
"""

from feast import Entity, ValueType

invoice = Entity(
    name="invoice_id",
    value_type=ValueType.STRING,
    description="The invoice document number (primary key, e.g. INV-2025-08842).",
)
