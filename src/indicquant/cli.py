"""IndicQuant CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from indicquant import __version__

app = typer.Typer(
    name="indicquant",
    help="Document intelligence harness for Indian paperwork. Extract, AutoOpt, eval, serve.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def chat(
    trace: bool = typer.Option(False, "--trace", help="Verbose bind/act/memory diff"),
    quiet: bool = typer.Option(False, "--quiet", help="Print only the answer"),
    local: bool = typer.Option(False, "--local", help="Ollama only; block cloud LLM URLs"),
) -> None:
    """Talk to the agent in the terminal. The model plans; tools run."""
    from indicquant.agent.chat import run_chat
    from indicquant.agent.llm import detect_llm

    try:
        llm = detect_llm(local=local)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    run_chat(llm, trace=trace, quiet=quiet, local=local)


@app.command()
def ask(
    message: str = typer.Argument(..., help="One user message; prints the turn and exits"),
    trace: bool = typer.Option(True, "--trace/--no-trace"),
    local: bool = typer.Option(False, "--local"),
    confirm: bool = typer.Option(False, "--confirm", help="Auto-confirm a pending booking"),
) -> None:
    """One shot. Does not take over the terminal."""
    from indicquant.agent.chat import build_agent, print_turn
    from indicquant.agent.llm import detect_llm

    try:
        llm = detect_llm(local=local)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    agent = build_agent(llm, auto_confirm=False, persist=False, local=local)
    turn = agent.act(message)
    print_turn(turn, verbose=trace)
    if confirm and turn.pending_id:
        turn = agent.confirm(turn.pending_id)
        print_turn(turn, verbose=trace)


@app.command()
def extract(
    sample: str | None = typer.Option(None, help="Golden-set id, e.g. pan_hi"),
    path: str | None = typer.Option(None, help="Text file of OCR / document content"),
) -> None:
    """Run the document pipeline: ingest → split → infer → validate → assemble."""
    from pathlib import Path

    from indicquant.harness.jobs import open_jobs

    store = open_jobs()
    if sample:
        job = store.submit_sample(sample)
    elif path:
        job = store.submit_text(Path(path).read_text(encoding="utf-8"))
    else:
        console.print("[red]pass --sample pan_hi or --path file.txt[/red]")
        raise typer.Exit(1)
    console.print(f"{job['id']}  {job['doc_type']}  {job['latency_ms']}ms  ok={job['ok']}")
    for stage in job.get("stages") or []:
        console.print(f"  {stage['name']:12} {stage['ms']:>7} ms")
    console.print_json(data=job.get("result"))


@app.command()
def autoopt(
    sample: str | None = typer.Option(None, help="Golden-set id, e.g. opt_lp2"),
    path: str | None = typer.Option(None, help="Text/LaTeX file of a formulation"),
) -> None:
    """Run AutoOpt: mer → compile checkpoint → BOBD/vertex solve. No GPU."""
    from pathlib import Path

    from indicquant.harness.jobs import open_jobs

    store = open_jobs()
    if sample:
        job = store.submit_sample(sample)
    elif path:
        job = store.submit_text(Path(path).read_text(encoding="utf-8"), pipeline="autoopt")
    else:
        console.print("[red]pass --sample opt_lp2 or --path model.txt[/red]")
        raise typer.Exit(1)
    console.print(f"{job['id']}  {job['doc_type']}  {job['latency_ms']}ms  ok={job['ok']}")
    for stage in job.get("stages") or []:
        mark = "ok" if stage.get("ok") else "FAIL"
        console.print(f"  {stage['name']:12} {stage['ms']:>7} ms  {mark}")
    sol = (job.get("result") or {}).get("solution")
    if sol:
        console.print(f"  method={sol.get('method')}  objective={sol.get('objective')}  x={sol.get('x')}")
    latex = (job.get("result") or {}).get("latex")
    if latex:
        console.print("[dim]M1 checkpoint (LaTeX/text)[/dim]")
        console.print(latex)
    if not job.get("ok"):
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    suite: str = typer.Option("documents", help="documents | autoopt | scans | agent"),
    path: str | None = typer.Option(None, help="JSON case file for the agent suite"),
) -> None:
    """Golden-set extraction (default) or the scripted agent booking suite."""
    from pathlib import Path

    if suite == "agent":
        from indicquant.eval.harness import run_eval

        report = run_eval(Path(path) if path else None)
        t = Table(title="indicquant eval — agent")
        t.add_column("id")
        t.add_column("ok")
        t.add_column("grounded")
        t.add_column("critic")
        t.add_column("ms")
        t.add_column("errors")
        for row in report["scores"]:
            t.add_row(
                row["id"],
                "yes" if row["ok"] else "NO",
                "yes" if row["grounded"] else "no",
                "caught" if row["hallucination_caught"] else "—",
                str(row["duration_ms"]),
                "; ".join(row["errors"])[:60],
            )
        console.print(t)
        console.print(
            f"task_completion_rate={report['task_completion_rate']}  "
            f"hallucination_catch_rate={report['hallucination_catch_rate']}  "
            f"latency_ms_mean={report['latency_ms_mean']}"
        )
        if any(not s["ok"] for s in report["scores"]):
            raise typer.Exit(1)
        return

    if suite == "autoopt":
        from indicquant.eval.autoopt import run_eval as run_autoopt_eval

        report = run_autoopt_eval()
        t = Table(title="indicquant eval — autoopt")
        t.add_column("id")
        t.add_column("ok")
        t.add_column("method")
        t.add_column("fields")
        t.add_column("ms")
        t.add_column("errors")
        for row in report["scores"]:
            t.add_row(
                row["id"],
                "yes" if row["ok"] else "NO",
                str(row.get("method") or ""),
                str(row["field_accuracy"]),
                str(row["latency_ms"]),
                "; ".join(row["errors"])[:60],
            )
        console.print(t)
        console.print(
            f"task_completion_rate={report['task_completion_rate']}  "
            f"field_accuracy_mean={report['field_accuracy_mean']}  "
            f"latency_ms_mean={report['latency_ms_mean']}"
        )
        if any(not s["ok"] for s in report["scores"]):
            raise typer.Exit(1)
        return

    if suite == "scans":
        from indicquant.eval.scans import run_eval as run_scan_eval

        report = run_scan_eval()
        t = Table(title="indicquant eval — scans (small MER)")
        t.add_column("id")
        t.add_column("ok")
        t.add_column("method")
        t.add_column("fields")
        t.add_column("png")
        t.add_column("errors")
        for row in report["scores"]:
            t.add_row(
                row["id"],
                "yes" if row["ok"] else "NO",
                str(row.get("method") or ""),
                str(row["field_accuracy"]),
                "yes" if row.get("has_png") else "no",
                "; ".join(row["errors"])[:60],
            )
        console.print(t)
        console.print(
            f"task_completion_rate={report['task_completion_rate']}  "
            f"field_accuracy_mean={report['field_accuracy_mean']}  "
            f"pngs={report['pngs']}/{report['cases']}"
        )
        if any(not s["ok"] for s in report["scores"]):
            raise typer.Exit(1)
        return

    from indicquant.eval.docs import run_eval as run_doc_eval

    report = run_doc_eval()
    t = Table(title="indicquant eval — documents")
    t.add_column("id")
    t.add_column("ok")
    t.add_column("type")
    t.add_column("fields")
    t.add_column("ms")
    t.add_column("errors")
    for row in report["scores"]:
        t.add_row(
            row["id"],
            "yes" if row["ok"] else "NO",
            str(row.get("doc_type") or ""),
            str(row["field_accuracy"]),
            str(row["latency_ms"]),
            "; ".join(row["errors"])[:60],
        )
    console.print(t)
    console.print(
        f"task_completion_rate={report['task_completion_rate']}  "
        f"field_accuracy_mean={report['field_accuracy_mean']}  "
        f"latency_ms_mean={report['latency_ms_mean']}"
    )
    if any(not s["ok"] for s in report["scores"]):
        raise typer.Exit(1)


@app.command()
def pending() -> None:
    """List confirm-gated side effects."""
    from indicquant.agent.pending import PendingStore

    rows = PendingStore().list(include_closed=True)[:20]
    if not rows:
        console.print("[dim]no pending actions[/dim]")
        return
    t = Table(title="pending actions")
    t.add_column("id", style="cyan")
    t.add_column("status")
    t.add_column("preview")
    t.add_column("expires")
    for action in rows:
        t.add_row(action.id, action.status, action.preview, action.expires_at)
    console.print(t)


@app.command()
def confirm(action_id: str) -> None:
    """Execute a pending booking by id (resumable across terminal sessions)."""
    from indicquant.agent.chat import build_agent
    from indicquant.agent.llm import detect_llm
    from indicquant.agent.pending import PendingStore

    store = PendingStore()
    try:
        action = store.get(action_id)
    except KeyError:
        console.print(f"[red]unknown action {action_id}[/red]")
        raise typer.Exit(1) from None
    try:
        llm = detect_llm()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    agent = build_agent(llm, auto_confirm=False, session_id=action.session_id, persist=True)
    try:
        agent.restore_pending(action)
        turn = agent.confirm(action_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    from indicquant.agent.chat import print_turn

    print_turn(turn, verbose=True)


@app.command()
def cancel(action_id: str) -> None:
    """Cancel a pending booking by id."""
    from indicquant.agent.pending import PendingStore

    try:
        action = PendingStore().mark(action_id, "cancelled")
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(f"cancelled {action.id}")


@app.command()
def traces() -> None:
    """Show persisted session traces under data/sessions/."""
    from indicquant.agent.session import load_trace_files

    rows = load_trace_files()
    if not rows:
        console.print("[dim]no sessions yet — run chat first[/dim]")
        return
    t = Table(title="Session traces")
    t.add_column("id", style="cyan")
    t.add_column("turns")
    t.add_column("lang")
    t.add_column("grounded")
    t.add_column("ms")
    t.add_column("last reply")
    for row in rows[:20]:
        t.add_row(
            row["id"],
            str(row["turns"]),
            str(row.get("language") or ""),
            "yes" if row.get("grounded") else "NO",
            str(row.get("duration_ms") or ""),
            str(row.get("last_reply") or ""),
        )
    console.print(t)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(7860),
) -> None:
    """Open the document console. Works without an LLM (extraction is code)."""
    from indicquant.agent.llm import detect_llm
    from indicquant.agent.serve import serve as run_serve

    llm = None
    try:
        llm = detect_llm()
    except RuntimeError as exc:
        console.print(f"[dim]docs-only (no chat LLM): {exc}[/dim]")
    console.print(f"[green]http://{host}:{port}[/green]")
    run_serve(host=host, port=port, llm=llm)


@app.command()
def targeting() -> None:
    """What this project ships."""
    t = Table(title="IndicQuant")
    t.add_column("Piece", style="cyan")
    t.add_column("What")
    t.add_row("Documents", "Indian paperwork: ingest → extract → validate → JSON")
    t.add_row("AutoOpt", "Checked loop: mer → compile → solve")
    t.add_row("Evals", "Golden sets for documents, AutoOpt, and labelled scans")
    console.print(t)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"indicquant {__version__}")


if __name__ == "__main__":
    app()
