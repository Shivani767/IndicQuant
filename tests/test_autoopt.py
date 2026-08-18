"""AutoOpt three-module harness — offline, no GPU."""

from __future__ import annotations

from indicquant.autoopt.m2 import compile_latex
from indicquant.autoopt.m3 import solve_program
from indicquant.autoopt.pipeline import looks_like_opt, run_autoopt
from indicquant.eval.autoopt import run_eval


def test_vertex_lp() -> None:
    program = compile_latex(
        "maximize 3*x + 4*y\nsubject to\nx + 2*y <= 8\n3*x + y <= 9\nx >= 0\ny >= 0"
    )
    sol = solve_program(program)
    assert sol["status"] == "optimal"
    assert sol["method"] == "vertex-lp"
    assert abs(sol["objective"] - 18.0) < 1e-4
    assert abs(sol["x"]["x"] - 2.0) < 1e-4
    assert abs(sol["x"]["y"] - 3.0) < 1e-4


def test_latex_checkpoint_and_min() -> None:
    program = compile_latex("\\min 3x + 2y\ns.t. x + y \\ge 4\nx \\ge 0\ny \\ge 0")
    sol = solve_program(program)
    assert abs(sol["objective"] - 8.0) < 1e-4
    assert abs(sol["x"]["y"] - 4.0) < 1e-4


def test_bobd_quadratic() -> None:
    program = compile_latex("minimize p - q^2\nsubject to\np + q <= 1\np >= 0\nq >= 0")
    sol = solve_program(program)
    assert sol["method"] == "bobd-grid"
    assert abs(sol["objective"] + 1.0) < 0.05
    assert abs(sol["x"]["q"] - 1.0) < 0.05


def test_incomplete_fails_at_compile() -> None:
    try:
        compile_latex("this is not a model")
    except ValueError as exc:
        assert "checkpoint" in str(exc).lower() or "min" in str(exc).lower()
    else:
        raise AssertionError("expected compile checkpoint failure")


def test_looks_like_opt_does_not_steal_documents() -> None:
    assert looks_like_opt("maximize 3*x + 4*y\nsubject to\nx <= 1")
    assert not looks_like_opt("TAX INVOICE\nGSTIN: 27ABCDE1234F1Z5\nTotal: 11800")


def test_pipeline_stages() -> None:
    out = run_autoopt(
        {
            "text": "maximize x + y\nsubject to\nx <= 1\ny <= 1\nx >= 0\ny >= 0",
            "pipeline": "autoopt",
        }
    )
    assert [s["name"] for s in out["stages"]] == [
        "ingest",
        "preprocess",
        "mer",
        "compile",
        "validate",
        "solve",
    ]
    assert out["ok"] is True
    assert abs(out["result"]["solution"]["objective"] - 2.0) < 1e-4
    assert out["result"]["latex"].startswith("maximize")


def test_compile_retry_inserts_st() -> None:
    out = run_autoopt(
        {
            "text": "minimize x + y\nx >= 1\ny >= 1\nx <= 5\ny <= 5",
            "pipeline": "autoopt",
        }
    )
    assert out["ok"] is True
    compile_stage = next(s for s in out["stages"] if s["name"] == "compile")
    assert compile_stage["detail"]["retried"] is True
    assert abs(out["result"]["solution"]["objective"] - 2.0) < 1e-4


def test_autoopt_golden_set() -> None:
    report = run_eval()
    assert report["cases"] >= 9
    failed = [s for s in report["scores"] if not s["ok"]]
    assert failed == [], failed
    assert report["task_completion_rate"] == 1.0
    assert any(s["id"] == "opt_incomplete" for s in report["scores"])
    assert any(s["id"] == "opt_infeasible" for s in report["scores"])


def test_joint_bounds_and_equality() -> None:
    program = compile_latex("maximize x + 2*y\ns.t. x + y <= 3, x, y >= 0")
    sol = solve_program(program)
    assert abs(sol["objective"] - 6.0) < 1e-4
    eq = compile_latex("minimize x + y\nsubject to\nx + y = 5\nx >= 1\ny >= 1")
    sol_eq = solve_program(eq)
    assert abs(sol_eq["objective"] - 5.0) < 1e-4


def test_job_dispatches_autoopt(tmp_path) -> None:
    from indicquant.harness.jobs import JobStore

    store = JobStore(tmp_path)
    job = store.submit_sample("opt_lp2")
    assert job["ok"] is True
    assert job["doc_type"] == "autoopt"
    assert job["result"]["solution"]["objective"] == 18.0
    assert any(s["name"] == "mer" for s in job["stages"])
