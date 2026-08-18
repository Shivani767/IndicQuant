"""M3: vertex LP (≤2 vars) or BOBD grid on one complicating variable."""

from __future__ import annotations

from typing import Any

from indicquant.autoopt.m2 import Constraint, Program

_EPS = 1e-8
_BOX = 1e6


class Infeasible(ValueError):
    pass


def evaluate(program: Program, point: dict[str, float]) -> float:
    val = 0.0
    for name, coef in program.objective.items():
        val += coef * point.get(name, 0.0)
    for name, coef in program.quadratic.items():
        val += coef * point.get(name, 0.0) ** 2
    return val


def feasible(program: Program, point: dict[str, float]) -> bool:
    for cons in program.constraints:
        val = sum(coef * point.get(name, 0.0) for name, coef in cons.coeffs.items())
        if cons.op == "<=" and val > cons.rhs + _EPS:
            return False
        if cons.op == ">=" and val < cons.rhs - _EPS:
            return False
        if cons.op == "=" and abs(val - cons.rhs) > _EPS:
            return False
    return True


def _round_point(point: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 6) for k, v in sorted(point.items())}


def _pick(program: Program, candidates: list[dict[str, float]]) -> dict[str, Any]:
    ok = [p for p in candidates if feasible(program, p)]
    if not ok:
        raise Infeasible("no feasible vertex")
    sense = 1.0 if program.sense == "min" else -1.0
    best = min(ok, key=lambda p: sense * evaluate(program, p))
    return {
        "x": _round_point(best),
        "objective": round(evaluate(program, best), 6),
        "status": "optimal",
        "method": "vertex-lp",
        "vertices_tested": len(ok),
    }


def _unary_bounds(program: Program, name: str) -> tuple[float, float]:
    lo, hi = -_BOX, _BOX
    for cons in program.constraints:
        if set(cons.coeffs) != {name}:
            continue
        a = cons.coeffs[name]
        if abs(a) < _EPS:
            continue
        bound = cons.rhs / a
        if cons.op == "<=":
            if a > 0:
                hi = min(hi, bound)
            else:
                lo = max(lo, bound)
        elif cons.op == ">=":
            if a > 0:
                lo = max(lo, bound)
            else:
                hi = min(hi, bound)
        elif cons.op == "=":
            lo = hi = bound
    return lo, hi


def _rest_ext(program: Program, cons: Constraint, skip: str, *, minimise: bool) -> float | None:
    total = 0.0
    for other, coef in cons.coeffs.items():
        if other == skip:
            continue
        lo, hi = _unary_bounds(program, other)
        take_lo = (coef > 0) if minimise else (coef < 0)
        if take_lo:
            if lo <= -_BOX / 2:
                return None
            total += coef * lo
        else:
            if hi >= _BOX / 2:
                return None
            total += coef * hi
    return total


def _var_bounds(program: Program, name: str) -> tuple[float, float]:
    lo, hi = _unary_bounds(program, name)
    for cons in program.constraints:
        a = cons.coeffs.get(name, 0.0)
        if abs(a) < _EPS or set(cons.coeffs) == {name}:
            continue
        if cons.op in {"<=", "="} and a > 0:
            rest = _rest_ext(program, cons, name, minimise=True)
            if rest is not None:
                hi = min(hi, (cons.rhs - rest) / a)
        if cons.op in {">=", "="} and a > 0:
            rest = _rest_ext(program, cons, name, minimise=False)
            if rest is not None:
                lo = max(lo, (cons.rhs - rest) / a)
    if hi >= _BOX / 2:
        hi = 20.0 if lo < 20.0 else lo + 20.0
    if lo <= -_BOX / 2:
        lo = 0.0 if hi > 0.0 else hi - 20.0
    return lo, hi


def _solve_1d(program: Program) -> dict[str, Any]:
    name = program.variables[0]
    lo, hi = _var_bounds(program, name)
    if lo > hi + _EPS:
        raise Infeasible(f"{name} bounds empty")
    candidates = [{name: lo}, {name: hi}]
    q = program.quadratic.get(name, 0.0)
    lin = program.objective.get(name, 0.0)
    if abs(q) > _EPS:
        vertex = -lin / (2.0 * q)
        if lo - _EPS <= vertex <= hi + _EPS:
            candidates.append({name: vertex})
    return _pick(program, candidates)


def _intersect(c1: Constraint, c2: Constraint, names: list[str]) -> dict[str, float] | None:
    x, y = names
    a1, b1 = c1.coeffs.get(x, 0.0), c1.coeffs.get(y, 0.0)
    a2, b2 = c2.coeffs.get(x, 0.0), c2.coeffs.get(y, 0.0)
    det = a1 * b2 - a2 * b1
    if abs(det) < _EPS:
        return None
    xv = (c1.rhs * b2 - c2.rhs * b1) / det
    yv = (a1 * c2.rhs - a2 * c1.rhs) / det
    if abs(xv) > _BOX or abs(yv) > _BOX:
        return None
    return {x: xv, y: yv}


def _solve_2d(program: Program) -> dict[str, Any]:
    names = program.variables
    cons = program.constraints
    candidates: list[dict[str, float]] = []
    for i, c1 in enumerate(cons):
        for c2 in cons[i + 1 :]:
            point = _intersect(c1, c2, names)
            if point is not None:
                candidates.append(point)
    return _pick(program, candidates)


def _fix(program: Program, name: str, value: float) -> Program:
    constraints = []
    for cons in program.constraints:
        a = cons.coeffs.get(name, 0.0)
        coeffs = {k: v for k, v in cons.coeffs.items() if k != name}
        constraints.append(Constraint(coeffs, cons.op, cons.rhs - a * value))
    return Program(
        sense=program.sense,
        objective={k: v for k, v in program.objective.items() if k != name},
        quadratic={k: v for k, v in program.quadratic.items() if k != name},
        constraints=constraints,
        variables=[n for n in program.variables if n != name],
        latex=program.latex,
    )


def _empty_ok(program: Program) -> bool:
    return feasible(program, {})


def bobd(program: Program, complicating: str, steps: int = 40) -> dict[str, Any]:
    """Grid the complicating variable; solve the remainder as LP (paper M3)."""
    lo, hi = _var_bounds(program, complicating)
    if lo > hi + _EPS:
        raise Infeasible("complicating variable has empty bounds")
    sense = 1.0 if program.sense == "min" else -1.0
    best_point: dict[str, float] | None = None
    best_obj = None
    tested = 0
    for i in range(steps + 1):
        q = lo + (hi - lo) * i / steps
        sub = _fix(program, complicating, q)
        try:
            if not sub.variables:
                if not _empty_ok(sub):
                    continue
                point = {complicating: q}
            else:
                inner = solve_program(sub, _allow_bobd=True)
                point = {**inner["x"], complicating: q}
            if not feasible(program, point):
                continue
        except Infeasible:
            continue
        tested += 1
        obj = evaluate(program, point)
        if best_obj is None or sense * obj < sense * best_obj:
            best_obj = obj
            best_point = point
    if best_point is None or best_obj is None:
        raise Infeasible("BOBD found no feasible sample")
    return {
        "x": _round_point(best_point),
        "objective": round(best_obj, 6),
        "status": "optimal",
        "method": "bobd-grid",
        "complicating": complicating,
        "grid": steps + 1,
        "feasible_samples": tested,
    }


def solve_program(program: Program, *, _allow_bobd: bool = True) -> dict[str, Any]:
    if program.quadratic and _allow_bobd:
        name = next(iter(program.quadratic))
        return bobd(program, name)
    n = len(program.variables)
    if n == 0:
        if not _empty_ok(program):
            raise Infeasible("empty program is infeasible")
        return {
            "x": {},
            "objective": 0.0,
            "status": "optimal",
            "method": "vertex-lp",
            "vertices_tested": 1,
        }
    if n == 1:
        return _solve_1d(program)
    if n == 2 and not program.quadratic:
        return _solve_2d(program)
    if _allow_bobd:
        return bobd(program, program.variables[-1])
    if n == 2:
        return _solve_2d(program)
    raise Infeasible("too many free variables for vertex LP")
