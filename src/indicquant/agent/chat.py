"""Interactive REPL — the model plans; the runtime only binds, acts, verifies."""

from __future__ import annotations

import uuid

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

from indicquant.agent.env import AgentEnv
from indicquant.agent.llm import LLM, detect_llm
from indicquant.agent.memory import EpisodicStore
from indicquant.agent.runtime import Agent, AgentTurn
from indicquant.agent.session import SessionLog, default_sessions_dir
from indicquant.agent.tools import user_tools
from indicquant.agent.trace import format_stages

console = Console()

DEGRADED_LOCAL = (
    "offline mode: local Ollama only. No cloud LLM. Lookup is the on-disk KB."
)


def build_agent(
    llm: LLM | None = None,
    *,
    auto_confirm: bool = True,
    persist: bool = True,
    session_id: str | None = None,
    local: bool = False,
) -> Agent:
    store = EpisodicStore()
    env = AgentEnv(tools=user_tools(store), scene=None)
    sid = session_id or uuid.uuid4().hex[:12]
    log = SessionLog(default_sessions_dir() / f"{sid}.jsonl") if persist else None
    return Agent(
        llm or detect_llm(local=local),
        env,
        memory=store,
        auto_confirm=auto_confirm,
        log=log,
        session_id=sid,
        local=local,
    )


def print_turn(turn: AgentTurn, *, quiet: bool = False, verbose: bool = False) -> None:
    if not quiet:
        for line in format_stages(turn, verbose=verbose):
            if line.startswith("[bind]"):
                console.print(f"[yellow]{line}[/yellow]")
            elif line.startswith("[critic]") or line.startswith("[pending]"):
                console.print(f"[magenta]{line}[/magenta]")
            elif line.startswith("[ground]"):
                console.print(f"[green]{line}[/green]")
            else:
                console.print(f"[dim]{line}[/dim]")
    badge = "[green]grounded[/green]" if turn.grounded else "[red]ungrounded[/red]"
    meta = f"{badge}  {turn.duration_ms:.0f}ms"
    console.print(
        Panel(Markdown(turn.reply or "_(empty reply)_"), title="agent", subtitle=meta, border_style="cyan")
    )
    if turn.pending_id:
        console.print(
            f"[magenta]confirm:[/magenta] uv run indicquant confirm {turn.pending_id}   "
            f"[dim]or /yes in this chat[/dim]"
        )


def _print_memory(agent: Agent) -> None:
    if not agent.memory.records:
        console.print("[dim]nothing remembered yet[/dim]")
        return
    for rec in agent.memory.records:
        flag = "verified" if rec.verified else rec.source
        console.print(f"  • {rec.fact}  [dim]{flag}[/dim]")


def run_chat(
    llm: LLM | None = None,
    *,
    trace: bool = False,
    quiet: bool = False,
    local: bool = False,
) -> None:
    agent = build_agent(llm or detect_llm(local=local), auto_confirm=False, local=local)
    last: AgentTurn | None = None
    console.print(
        "[bold]IndicQuant[/bold]  [dim]/yes /cancel /memory /trace /reset /quit[/dim]\n"
        f"[dim]backend: {agent.llm.name}  session: {agent.session_id}[/dim]"
    )
    if local:
        console.print(f"[yellow]{DEGRADED_LOCAL}[/yellow]")
    console.print()
    while True:
        try:
            text = console.input("[bold green]you[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        if not text:
            continue
        if text in {"/quit", "/exit", ":q"}:
            console.print("[dim]bye[/dim]")
            return
        if text == "/reset":
            agent.reset()
            last = None
            console.print("[dim]session cleared[/dim]")
            continue
        if text == "/memory":
            _print_memory(agent)
            continue
        if text == "/trace":
            if last is None:
                console.print("[dim]no turn yet[/dim]")
            else:
                for line in format_stages(last, verbose=True):
                    console.print(line)
            continue
        if text in {"/yes", "/confirm", "y"} and agent._pending:
            with Live(Spinner("dots", text="confirming"), console=console, transient=True):
                last = agent.confirm()
            print_turn(last, quiet=quiet, verbose=trace)
            continue
        if text in {"/cancel", "/no"} and agent._pending:
            last = agent.cancel()
            print_turn(last, quiet=quiet, verbose=trace)
            continue
        with Live(Spinner("dots", text="thinking"), console=console, transient=True):
            last = agent.act(text)
        print_turn(last, quiet=quiet, verbose=trace)
