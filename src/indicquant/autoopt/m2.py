"""M2: LaTeX/plain math → structured LP. Incomplete models fail here."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

_SENSE = re.compile(r"\\?(min(?:imize)?|max(?:imize)?)\.?", re.I)
_LE = re.compile(r"\\le(?:q)?|<=|≤")
_GE = re.compile(r"\\ge(?:q)?|>=|≥")
_ST = re.compile(r"(?:s\.t\.|subject\s+to)", re.I)


@dataclass
class Constraint:
    coeffs: dict[str, float]
    op: str  # <= >= =
    rhs: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Program:
    sense: str
    objective: dict[str, float]
    quadratic: dict[str, float]
    constraints: list[Constraint] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    latex: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sense": self.sense,
            "objective": self.objective,
            "quadratic": self.quadratic,
            "constraints": [c.to_dict() for c in self.constraints],
            "variables": self.variables,
            "latex": self.latex,
        }


def _clean(latex: str) -> str:
    text = latex
    for a, b in (
        (r"\min", "min"),
        (r"\max", "max"),
        (r"\leq", "<="),
        (r"\le", "<="),
        (r"\geq", ">="),
        (r"\ge", ">="),
        (r"\cdot", "*"),
        (r"\times", "*"),
        ("$", ""),
        (r"\,", " "),
        (r"\ ", " "),
        (r"\left", ""),
        (r"\right", ""),
        ("{", ""),
        ("}", ""),
    ):
        text = text.replace(a, b)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return text


def _term_map(expr: str) -> tuple[dict[str, float], dict[str, float]]:
    linear: dict[str, float] = {}
    quad: dict[str, float] = {}
    body = expr.replace(" ", "").replace("**", "^")
    body = re.sub(r"([a-zA-Z]\w*)\^2", r"\1*\1", body)
    body = re.sub(r"([a-zA-Z]\w*)²", r"\1*\1", body)
    if not body:
        return linear, quad
    if not body.startswith(("+", "-")):
        body = "+" + body
    parts = re.findall(r"[+-][^+-]+", body)
    if not parts:
        raise ValueError(f"cannot parse expression {expr!r}")
    for part in parts:
        sign = -1.0 if part.startswith("-") else 1.0
        token = part[1:].replace("*", "")
        msq = re.fullmatch(r"(\d*\.?\d*)([a-zA-Z]\w*)\2", token)
        if msq:
            coef = float(msq.group(1) or "1") * sign
            name = msq.group(2)
            quad[name] = quad.get(name, 0.0) + coef
            continue
        m = re.fullmatch(r"(\d*\.?\d*)([a-zA-Z]\w*)", token)
        if m:
            coef = float(m.group(1) or "1") * sign
            name = m.group(2)
            linear[name] = linear.get(name, 0.0) + coef
            continue
        if re.fullmatch(r"\d+\.?\d*", token):
            continue
        raise ValueError(f"cannot parse term {part!r}")
    return linear, quad


def _constraint(line: str) -> Constraint:
    if ">=" in line:
        op = ">="
        left, right = line.split(">=", 1)
    elif "<=" in line:
        op = "<="
        left, right = line.split("<=", 1)
    elif "=" in line:
        op = "="
        left, right = line.split("=", 1)
    else:
        raise ValueError(f"not a constraint: {line}")
    lin, quad = _term_map(left)
    if quad:
        raise ValueError("quadratic constraints are not compiled")
    rhs = float(right.strip())
    return Constraint(lin, op, rhs)


def _constraint_lines(block: str) -> list[str]:
    lines: list[str] = []
    for chunk in re.split(r"[;\n]", block):
        chunk = chunk.strip().rstrip(",")
        if not chunk:
            continue
        pieces = re.split(r"(?<=\d)\s*,\s*(?=[a-zA-Z].*[<>=])", chunk)
        lines.extend(p.strip() for p in pieces if p.strip())
    return lines


def _constraints_from(line: str) -> list[Constraint]:
    joint = re.match(
        r"^([a-zA-Z]\w*(?:\s*,\s*[a-zA-Z]\w*)+)\s*(>=|<=|=)\s*([-+]?\d+(?:\.\d+)?)$",
        line.strip(),
    )
    if joint:
        op, rhs = joint.group(2), float(joint.group(3))
        return [Constraint({name.strip(): 1.0}, op, rhs) for name in joint.group(1).split(",")]
    return [_constraint(line)]


def _retry_insert_st(latex: str) -> str:
    lines = [ln.strip() for ln in latex.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return latex
    return lines[0] + "\ns.t.\n" + "\n".join(lines[1:])


def compile_latex(latex: str) -> Program:
    raw = _clean(latex)
    raw = _LE.sub("<=", raw)
    raw = _GE.sub(">=", raw)
    sense_m = _SENSE.search(raw)
    if not sense_m:
        raise ValueError("M2 checkpoint: no min/max — formulation incomplete")
    sense = "min" if sense_m.group(1).lower().startswith("min") else "max"
    rest = raw[sense_m.end() :]
    split = _ST.split(rest, maxsplit=1)
    obj_expr = split[0].strip(" \n:.")
    cons_block = split[1] if len(split) > 1 else ""
    linear, quad = _term_map(obj_expr)
    constraints: list[Constraint] = []
    for line in _constraint_lines(cons_block):
        if line.lower() in {"s.t.", "st"}:
            continue
        constraints.extend(_constraints_from(line))
    names = sorted(set(linear) | set(quad) | {n for c in constraints for n in c.coeffs})
    if not names:
        raise ValueError("M2 checkpoint: no decision variables")
    if not constraints:
        raise ValueError("M2 checkpoint: no constraints")
    return Program(
        sense=sense,
        objective=linear,
        quadratic=quad,
        constraints=constraints,
        variables=names,
        latex=latex.strip(),
    )


def compile_latex_retry(latex: str) -> tuple[Program, bool]:
    """Paper: not every M1 transcript is executable — retry compile once."""
    try:
        return compile_latex(latex), False
    except ValueError:
        return compile_latex(_retry_insert_st(latex)), True
