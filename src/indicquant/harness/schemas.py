"""English field names; values may be Indic."""

from __future__ import annotations

from typing import Any

SCHEMAS: dict[str, dict[str, Any]] = {
    "pan": {
        "title": "PAN card",
        "required": ["pan", "name"],
        "fields": ["pan", "name", "father_name", "dob"],
    },
    "aadhaar": {
        "title": "Aadhaar (sample IDs only)",
        "required": ["aadhaar", "name"],
        "fields": ["aadhaar", "name", "dob", "gender"],
    },
    "gst_invoice": {
        "title": "GST tax invoice",
        "required": ["gstin", "invoice_no", "total"],
        "fields": ["gstin", "invoice_no", "date", "bill_to", "taxable", "cgst", "sgst", "igst", "total"],
    },
    "bank_statement": {
        "title": "Bank statement",
        "required": ["ifsc", "account"],
        "fields": ["bank", "ifsc", "account", "period", "opening", "closing"],
    },
    "insurance": {
        "title": "Insurance policy",
        "required": ["policy_no", "premium"],
        "fields": ["insurer", "policy_no", "premium", "sum_assured"],
    },
    "driving_licence": {
        "title": "Driving licence (sample IDs only)",
        "required": ["dl_no", "name"],
        "fields": ["dl_no", "name", "dob"],
    },
}

DOC_TYPES = tuple(SCHEMAS)
