"""Schema extractors: regex + native-digit bind + arithmetic checks."""

from __future__ import annotations

import re
from typing import Any

from indicquant.agent.script import ascii_digits
from indicquant.harness.schemas import SCHEMAS

PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
GSTIN_RE = re.compile(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z])\b")
AADHAAR_RE = re.compile(r"\b(\d{4}[\s-]?\d{4}[\s-]?\d{4})\b")
IFSC_RE = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
DL_RE = re.compile(r"\b([A-Z]{2}\d{2}[\s-]?\d{11})\b")
INVOICE_RE = re.compile(r"(?:invoice\s*(?:no\.?|number|#)|चालान)\s*[:.\-]?\s*([A-Z0-9][\w\-\/]+)", re.I)
POLICY_RE = re.compile(r"(?:policy|पॉलिसी)\s*(?:no\.?|number|#)?\s*[:.\-]?\s*([A-Z0-9][\w\-]*)", re.I)
ACCOUNT_RE = re.compile(r"(?:account|a/c|खाता)\s*(?:no\.?|number)?\s*[:.\-]?\s*(\d{9,18})", re.I)
DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
MONEY_RE = re.compile(
    r"(?:₹|rs\.?|inr)?\s*([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)",
    re.I,
)

_NAME_LABELS = ("नाम", "name", "பெயர்", "ନାମ")
_FATHER_LABELS = ("पिता", "father", "father's name")
_BANK_LABELS = ("bank", "बैंक")
_FEMALE = re.compile(r"(female|महिला|பெண்)", re.I)
_MALE = re.compile(r"(male|पुरुष|ஆண்)", re.I)


def _norm(text: str) -> str:
    return ascii_digits(text).replace("\u00a0", " ")


def _money(text: str, *labels: str) -> float | None:
    blob = _norm(text)
    for label in labels:
        loc = re.compile(rf"(?:^|[\n\s]){re.escape(label)}(?=\s|%|:|\d)", re.I)
        match = loc.search(blob)
        if not match:
            continue
        idx = match.start()
        window = blob[idx : idx + 96]
        if ":" in window:
            after = window.split(":", 1)[1]
            amount = MONEY_RE.search(after)
            if amount:
                return float(amount.group(1).replace(",", ""))
        matches = list(MONEY_RE.finditer(window))
        for amount in reversed(matches):
            value = float(amount.group(1).replace(",", ""))
            if value > 28:
                return value
    return None


def _labeled(text: str, labels: tuple[str, ...]) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        for label in labels:
            if label.lower() not in lower and label not in line:
                continue
            for sep in (":", "/", "-", "–"):
                if sep in line:
                    rhs = line.split(sep, 1)[1].strip()
                    rhs = re.sub(r"^(name|नाम|பெயர்)\s*", "", rhs, flags=re.I).strip()
                    if rhs and rhs.lower() not in {item.lower() for item in labels}:
                        return rhs
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return None


def extract_pan(text: str) -> dict[str, Any]:
    blob = _norm(text).upper()
    pan = PAN_RE.search(blob)
    dob = DATE_RE.search(_norm(text))
    return {
        "pan": pan.group(1) if pan else None,
        "name": _labeled(text, _NAME_LABELS),
        "father_name": _labeled(text, _FATHER_LABELS),
        "dob": dob.group(1) if dob else None,
    }


def extract_aadhaar(text: str) -> dict[str, Any]:
    blob = _norm(text)
    num = AADHAAR_RE.search(blob)
    aadhaar = None
    if num:
        digits = re.sub(r"\D", "", num.group(1))
        if len(digits) == 12:
            aadhaar = f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"
    gender = None
    if _FEMALE.search(blob):
        gender = "F"
    elif _MALE.search(blob):
        gender = "M"
    dob = DATE_RE.search(blob)
    return {
        "aadhaar": aadhaar,
        "name": _labeled(text, _NAME_LABELS),
        "dob": dob.group(1) if dob else None,
        "gender": gender,
    }


def extract_gst(text: str) -> dict[str, Any]:
    blob = _norm(text)
    gstin = GSTIN_RE.search(blob.upper())
    inv = INVOICE_RE.search(blob)
    date = DATE_RE.search(blob)
    return {
        "gstin": gstin.group(1) if gstin else None,
        "invoice_no": inv.group(1) if inv else None,
        "date": date.group(1) if date else None,
        "bill_to": _labeled(text, ("bill to", "billed to", "ग्राहक", "customer")),
        "taxable": _money(text, "taxable", "taxable value", "कर योग्य"),
        "cgst": _money(text, "cgst"),
        "sgst": _money(text, "sgst"),
        "igst": _money(text, "igst"),
        "total": _money(text, "total", "grand total", "कुल", "invoice total"),
    }


def extract_bank(text: str) -> dict[str, Any]:
    blob = _norm(text)
    ifsc = IFSC_RE.search(blob.upper())
    acct = ACCOUNT_RE.search(blob)
    bank = _labeled(text, _BANK_LABELS)
    if bank is None:
        for line in text.splitlines():
            if "bank" in line.lower() or "बैंक" in line:
                bank = line.strip()
                break
    period = None
    for line in text.splitlines():
        if "period" in line.lower() or "अवधि" in line:
            period = line.split(":", 1)[-1].strip()
            break
    return {
        "bank": bank,
        "ifsc": ifsc.group(1) if ifsc else None,
        "account": acct.group(1) if acct else None,
        "period": period,
        "opening": _money(text, "opening", "opening balance"),
        "closing": _money(text, "closing", "closing balance"),
    }


def extract_insurance(text: str) -> dict[str, Any]:
    blob = _norm(text)
    pol = POLICY_RE.search(blob)
    insurer = None
    for line in text.splitlines():
        if any(tok in line.lower() for tok in ("life", "insurance", "बीमा", "sbi", "lic", "hdfc")):
            insurer = line.strip()
            break
    return {
        "insurer": insurer,
        "policy_no": pol.group(1) if pol else None,
        "premium": _money(text, "premium", "प्रीमियम"),
        "sum_assured": _money(text, "sum assured", "बीमा राशि"),
    }


def extract_dl(text: str) -> dict[str, Any]:
    blob = _norm(text).upper()
    dl = DL_RE.search(blob)
    dob = DATE_RE.search(_norm(text))
    number = None
    if dl:
        number = re.sub(r"[\s-]", "", dl.group(1))
    return {
        "dl_no": number,
        "name": _labeled(text, _NAME_LABELS),
        "dob": dob.group(1) if dob else None,
    }


EXTRACTORS = {
    "pan": extract_pan,
    "aadhaar": extract_aadhaar,
    "gst_invoice": extract_gst,
    "bank_statement": extract_bank,
    "insurance": extract_insurance,
    "driving_licence": extract_dl,
}


def route_doc_type(text: str) -> str:
    blob = _norm(text)
    scores: dict[str, int] = {key: 0 for key in EXTRACTORS}
    if PAN_RE.search(blob.upper()) or "permanent account" in blob.lower() or "आयकर" in text:
        scores["pan"] += 3
    if AADHAAR_RE.search(blob) or "aadhaar" in blob.lower() or "आधार" in text:
        scores["aadhaar"] += 3
    if GSTIN_RE.search(blob.upper()) or "tax invoice" in blob.lower() or "gstin" in blob.lower():
        scores["gst_invoice"] += 3
    if IFSC_RE.search(blob.upper()) or "opening balance" in blob.lower():
        scores["bank_statement"] += 3
    if "sum assured" in blob.lower() or "premium" in blob.lower() or "पॉलिसी" in text:
        scores["insurance"] += 3
    if DL_RE.search(blob.upper()) or "driving licence" in blob.lower() or "driving license" in blob.lower():
        scores["driving_licence"] += 3
    filled = {key: sum(1 for v in fn(text).values() if v not in (None, "")) for key, fn in EXTRACTORS.items()}
    return max(EXTRACTORS, key=lambda k: (scores[k], filled[k]))


def confidence(fields: dict[str, Any], doc_type: str) -> dict[str, float]:
    schema = SCHEMAS[doc_type]
    return {key: 1.0 if fields.get(key) not in (None, "") else 0.0 for key in schema["fields"]}


def _gst_identity(fields: dict[str, Any]) -> bool:
    taxable, total = fields.get("taxable"), fields.get("total")
    if taxable is None or total is None:
        return True
    tax = sum(v for v in (fields.get("cgst"), fields.get("sgst"), fields.get("igst")) if v is not None)
    return abs(float(taxable) + tax - float(total)) <= 0.51


def validate(fields: dict[str, Any], doc_type: str) -> dict[str, Any]:
    schema = SCHEMAS[doc_type]
    missing = [key for key in schema["required"] if fields.get(key) in (None, "")]
    errors: list[str] = list(missing)
    pan = fields.get("pan")
    if pan and not PAN_RE.fullmatch(str(pan)):
        errors.append("pan_format")
    gstin = fields.get("gstin")
    if gstin and not GSTIN_RE.fullmatch(str(gstin)):
        errors.append("gstin_format")
    aadhaar = fields.get("aadhaar")
    if aadhaar and len(re.sub(r"\D", "", str(aadhaar))) != 12:
        errors.append("aadhaar_format")
    ifsc = fields.get("ifsc")
    if ifsc and not IFSC_RE.fullmatch(str(ifsc)):
        errors.append("ifsc_format")
    dl = fields.get("dl_no")
    if dl and not DL_RE.fullmatch(str(dl)) and not re.fullmatch(r"[A-Z]{2}\d{13}", str(dl)):
        errors.append("dl_format")
    if doc_type == "gst_invoice" and not _gst_identity(fields):
        errors.append("gst_total_mismatch")
    return {"ok": not errors, "missing": missing, "errors": errors}


def extract(text: str, doc_type: str | None = None) -> dict[str, Any]:
    kind = doc_type or route_doc_type(text)
    fn = EXTRACTORS[kind]
    fields = fn(text)
    report = validate(fields, kind)
    if not report["ok"]:
        fields = fn(_norm(text))
        report = validate(fields, kind)
    return {
        "doc_type": kind,
        "fields": fields,
        "confidence": confidence(fields, kind),
        "validation": report,
    }
