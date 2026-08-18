"""Document golden-set eval."""

from __future__ import annotations

from indicquant.eval.docs import run_eval


def test_document_golden_set() -> None:
    report = run_eval()
    assert report["cases"] >= 10
    failed = [s for s in report["scores"] if not s["ok"]]
    assert failed == [], failed
    assert report["task_completion_rate"] == 1.0
    assert report["field_accuracy_mean"] == 1.0


def test_two_page_fanout() -> None:
    from indicquant.harness.pipeline import run_pipeline
    from indicquant.harness.samples import get_case

    out = run_pipeline(get_case("gst_two_page"))
    names = [s["name"] for s in out["stages"]]
    assert names == ["ingest", "split", "preprocess", "infer", "validate", "assemble"]
    assert out["stages"][1]["detail"]["fanout"] == 2
    assert out["result"]["fields"]["gstin"] == "29AAAAA0000A1Z5"
    assert out["result"]["fields"]["total"] == 2360.0
    assert "cpu_ms" in out["cost"]["unit"]


def test_gst_tax_identity() -> None:
    from indicquant.harness.extract import extract

    out = extract(
        "TAX INVOICE GSTIN: 27ABCDE1234F1Z5 Invoice No: INV-1 Date: 01/01/2026 "
        "Taxable: 100 CGST: 9 SGST: 9 Total: 999",
        "gst_invoice",
    )
    assert out["validation"]["ok"] is False
    assert "gst_total_mismatch" in out["validation"]["errors"]


def test_dl_and_native_digits() -> None:
    from indicquant.harness.pipeline import run_pipeline
    from indicquant.harness.samples import get_case

    dl = run_pipeline(get_case("dl_en"))
    assert dl["result"]["fields"]["dl_no"] == "MH1420110012345"
    pan = run_pipeline(get_case("pan_hi_digits"))
    assert pan["result"]["fields"]["dob"] == "12/03/1994"
    assert pan["result"]["fields"]["pan"] == "HJKLM2345N"


def test_job_store_roundtrip(tmp_path) -> None:
    from indicquant.harness.jobs import JobStore

    store = JobStore(tmp_path)
    job = store.submit_sample("pan_hi")
    assert job["ok"] is True
    assert job["result"]["fields"]["pan"] == "ABCDE1234F"
    loaded = store.get(job["id"])
    assert loaded["id"] == job["id"]
    assert any(j["id"] == job["id"] for j in store.list())
