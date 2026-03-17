from __future__ import annotations

import ast
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple


# -----------------------------
# Formula model
# -----------------------------
@dataclass
class FormulaSpec:
    key: str
    label: str
    kind: str  # risk | measure | metric
    expression: str
    description: str
    dependencies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FORMULA_LIBRARY: Dict[str, FormulaSpec] = {
    "risk_pressure_default": FormulaSpec(
        key="risk_pressure_default",
        label="Risk Pressure Default",
        kind="risk",
        expression="strategy_effect - impact_score - 0.8 * risk_score + 0.05 * metric_total",
        description="Standardwirkung eines negativen Ereignisses auf die Strategie. Berücksichtigt Direkteffekt, Einflussstärke, Risikostärke und aggregierte Kennzahlen.",
        dependencies=["strategy_effect", "impact_score", "risk_score", "metric_total"],
    ),
    "risk_supply_shock": FormulaSpec(
        key="risk_supply_shock",
        label="Supply Shock",
        kind="risk",
        expression="strategy_effect - 1.15 * impact_score - 1.1 * risk_score - 0.06 * abs(Pegel) + 0.03 * Verbrauch - 0.04 * Reservekapazität",
        description="Geeignet für Versorgungsschocks, Trockenphasen oder Ausfälle mit Pegel- und Reserveeffekten.",
        dependencies=["strategy_effect", "impact_score", "risk_score", "Pegel", "Verbrauch", "Reservekapazität"],
    ),
    "measure_stabilization_default": FormulaSpec(
        key="measure_stabilization_default",
        label="Measure Stabilization Default",
        kind="measure",
        expression="activation * (0.9 * effectiveness_score + 0.75 * strategy_alignment + 0.04 * metric_total)",
        description="Standardwirkung einer Maßnahme. Berücksichtigt Wirksamkeit, Strategiebeitrag, Aktivierungsgrad und Kennzahlen.",
        dependencies=["activation", "effectiveness_score", "strategy_alignment", "metric_total"],
    ),
    "measure_capacity_build": FormulaSpec(
        key="measure_capacity_build",
        label="Capacity Build",
        kind="measure",
        expression="activation * (0.8 * effectiveness_score + 0.9 * strategy_alignment + 0.08 * Reservekapazität + 0.05 * Prognosegüte)",
        description="Geeignet für Kapazitätsaufbau, Prognoseverbesserung und operative Absicherung.",
        dependencies=["activation", "effectiveness_score", "strategy_alignment", "Reservekapazität", "Prognosegüte"],
    ),
    "metric_linear": FormulaSpec(
        key="metric_linear",
        label="Metric Linear",
        kind="metric",
        expression="0.12 * metric_value",
        description="Lineare Standardwirkung einer Kennzahl.",
        dependencies=["metric_value"],
    ),
    "metric_inverse_risk": FormulaSpec(
        key="metric_inverse_risk",
        label="Metric Inverse Risk",
        kind="metric",
        expression="0.08 * metric_value - 0.03 * risk_score",
        description="Kennzahl mit zusätzlicher Abschwächung durch hohes Risiko.",
        dependencies=["metric_value", "risk_score"],
    ),
}

EVENT_METRIC_FORMULAS: Dict[str, str] = {
    "default": "metric_linear",
    "Pegel": "metric_inverse_risk",
    "Verbrauch": "metric_linear",
    "Druckindex": "metric_inverse_risk",
}

MEASURE_METRIC_FORMULAS: Dict[str, str] = {
    "default": "metric_linear",
    "Reservekapazität": "metric_linear",
    "Prognosegüte": "metric_linear",
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Compare,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.IfExp,
)


class FormulaError(Exception):
    pass


def available_formulas(kind: str | None = None) -> List[FormulaSpec]:
    specs = list(FORMULA_LIBRARY.values())
    if kind is None:
        return sorted(specs, key=lambda s: (s.kind, s.label))
    return sorted([s for s in specs if s.kind == kind], key=lambda s: s.label)


def get_formula_spec(key: str) -> FormulaSpec:
    if key not in FORMULA_LIBRARY:
        raise FormulaError(f"Unbekannte Formel: {key}")
    return FORMULA_LIBRARY[key]


def _validate_expression(expr: str) -> ast.AST:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"Syntaxfehler in Formel: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise FormulaError(f"Nicht erlaubter Ausdruckstyp: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
                raise FormulaError("Nur freigegebene Funktionen sind erlaubt: abs, min, max, round, sqrt, log, exp")
    return tree


def evaluate_expression(expression: str, context: Mapping[str, Any]) -> float:
    tree = _validate_expression(expression)
    names = {**SAFE_FUNCTIONS, **context}
    try:
        value = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, names)
    except Exception as exc:
        raise FormulaError(f"Fehler bei der Formelauswertung: {exc}") from exc
    try:
        return float(value)
    except Exception as exc:
        raise FormulaError(f"Formel lieferte keinen numerischen Wert: {value}") from exc


def register_formula(spec: FormulaSpec) -> None:
    FORMULA_LIBRARY[spec.key] = spec


def build_context_for_event(event: Any, scenario: Any) -> Dict[str, float]:
    metrics = dict(getattr(event, "metrics", {}) or {})
    direction = getattr(event, "direction", "negative")
    direction_factor = -1.0 if direction == "negative" else 0.6 if direction == "positive" else 0.0
    ctx: Dict[str, float] = {
        "impact_score": float(getattr(event, "impact_score", 0.0)),
        "risk_score": float(getattr(event, "risk_score", 0.0)),
        "strategy_effect": float(getattr(event, "strategy_effect", 0.0)),
        "baseline_trend": float(getattr(scenario, "baseline_trend", 0.0)),
        "strategy_sensitivity": float(getattr(scenario, "strategy_sensitivity", 1.0)),
        "current_value": float(getattr(scenario, "current_value", 0.0)),
        "target_value": float(getattr(scenario, "target_value", 0.0)),
        "direction_factor": float(direction_factor),
        "metric_total": float(sum(float(v) for v in metrics.values())),
    }
    for k, v in metrics.items():
        ctx[str(k)] = float(v)
    return ctx


def build_context_for_measure(measure: Any, scenario: Any) -> Dict[str, float]:
    metrics = dict(getattr(measure, "metrics", {}) or {})
    activation = 1.0 if getattr(measure, "status", "planned") in {"active", "completed"} else 0.45
    ctx: Dict[str, float] = {
        "effectiveness_score": float(getattr(measure, "effectiveness_score", 0.0)),
        "strategy_alignment": float(getattr(measure, "strategy_alignment", 0.0)),
        "activation": float(activation),
        "baseline_trend": float(getattr(scenario, "baseline_trend", 0.0)),
        "strategy_sensitivity": float(getattr(scenario, "strategy_sensitivity", 1.0)),
        "current_value": float(getattr(scenario, "current_value", 0.0)),
        "target_value": float(getattr(scenario, "target_value", 0.0)),
        "metric_total": float(sum(float(v) for v in metrics.values())),
    }
    for k, v in metrics.items():
        ctx[str(k)] = float(v)
    return ctx


def _metric_formula_key(metric_name: str, kind: str) -> str:
    mapping = EVENT_METRIC_FORMULAS if kind == "event" else MEASURE_METRIC_FORMULAS
    return mapping.get(metric_name, mapping.get("default", "metric_linear"))


def _evaluate_metric_effects(metrics: Mapping[str, float], context: Dict[str, float], kind: str) -> Tuple[float, List[Dict[str, Any]]]:
    total = 0.0
    rows: List[Dict[str, Any]] = []
    for metric_name, metric_value in metrics.items():
        formula_key = _metric_formula_key(str(metric_name), kind)
        spec = get_formula_spec(formula_key)
        local_context = dict(context)
        local_context["metric_name"] = str(metric_name)
        local_context["metric_value"] = float(metric_value)
        value = evaluate_expression(spec.expression, local_context)
        rows.append({
            "metric_name": str(metric_name),
            "metric_value": float(metric_value),
            "formula_key": formula_key,
            "expression": spec.expression,
            "effect": float(value),
        })
        total += float(value)
    return total, rows


def event_strategy_delta(event: Any, scenario: Any) -> float:
    context = build_context_for_event(event, scenario)
    formula_key = getattr(event, "formula_key", None) or "risk_pressure_default"
    spec = get_formula_spec(formula_key)
    base = evaluate_expression(spec.expression, context)
    metrics = dict(getattr(event, "metrics", {}) or {})
    metric_total, _ = _evaluate_metric_effects(metrics, context, "event")
    return (base + metric_total) * float(getattr(scenario, "strategy_sensitivity", 1.0))


def measure_strategy_delta(measure: Any, scenario: Any) -> float:
    context = build_context_for_measure(measure, scenario)
    formula_key = getattr(measure, "formula_key", None) or "measure_stabilization_default"
    spec = get_formula_spec(formula_key)
    base = evaluate_expression(spec.expression, context)
    metrics = dict(getattr(measure, "metrics", {}) or {})
    metric_total, _ = _evaluate_metric_effects(metrics, context, "measure")
    return (base + metric_total) * float(getattr(scenario, "strategy_sensitivity", 1.0))


def event_accuracy_delta(event: Any) -> float:
    direction = getattr(event, "direction", "negative")
    impact = float(getattr(event, "impact_score", 0.0))
    risk = float(getattr(event, "risk_score", 0.0))
    direct = float(getattr(event, "strategy_effect", 0.0))
    if direction == "negative":
        return -(impact * 1.15 + risk * 0.8 + max(0.0, -direct) * 0.4)
    if direction == "positive":
        return impact * 0.8 + max(0.0, direct) * 0.55
    return direct * 0.2


def measure_accuracy_delta(measure: Any) -> float:
    activation = 1.0 if getattr(measure, "status", "planned") in {"active", "completed"} else 0.45
    eff = float(getattr(measure, "effectiveness_score", 0.0))
    align = float(getattr(measure, "strategy_alignment", 0.0))
    return (eff * 0.9 + max(0.0, align) * 0.55) * activation


def explain_event(event: Any, scenario: Any) -> Dict[str, Any]:
    context = build_context_for_event(event, scenario)
    formula_key = getattr(event, "formula_key", None) or "risk_pressure_default"
    spec = get_formula_spec(formula_key)
    base = evaluate_expression(spec.expression, context)
    metric_total, metric_rows = _evaluate_metric_effects(dict(getattr(event, "metrics", {}) or {}), context, "event")
    total = (base + metric_total) * float(getattr(scenario, "strategy_sensitivity", 1.0))
    return {
        "type": "event",
        "formula_key": formula_key,
        "expression": spec.expression,
        "description": spec.description,
        "dependencies": spec.dependencies,
        "context": context,
        "base_effect": float(base),
        "metric_effect": float(metric_total),
        "metric_rows": metric_rows,
        "scenario_sensitivity": float(getattr(scenario, "strategy_sensitivity", 1.0)),
        "total_effect": float(total),
    }


def explain_measure(measure: Any, scenario: Any) -> Dict[str, Any]:
    context = build_context_for_measure(measure, scenario)
    formula_key = getattr(measure, "formula_key", None) or "measure_stabilization_default"
    spec = get_formula_spec(formula_key)
    base = evaluate_expression(spec.expression, context)
    metric_total, metric_rows = _evaluate_metric_effects(dict(getattr(measure, "metrics", {}) or {}), context, "measure")
    total = (base + metric_total) * float(getattr(scenario, "strategy_sensitivity", 1.0))
    return {
        "type": "measure",
        "formula_key": formula_key,
        "expression": spec.expression,
        "description": spec.description,
        "dependencies": spec.dependencies,
        "context": context,
        "base_effect": float(base),
        "metric_effect": float(metric_total),
        "metric_rows": metric_rows,
        "scenario_sensitivity": float(getattr(scenario, "strategy_sensitivity", 1.0)),
        "total_effect": float(total),
    }


def build_strategy_series(scenario: Any) -> Tuple[List[float], List[float], List[float], List[float]]:
    checkpoints = [0.0]
    baseline_values = [float(scenario.current_value)]
    strategy_values = [float(scenario.current_value)]
    accuracy_values = [float(scenario.baseline_accuracy)]

    baseline_running = float(scenario.current_value)
    strategy_running = float(scenario.current_value)
    accuracy_running = float(scenario.baseline_accuracy)
    last_week = 0.0

    objects = [(float(e.position), "event", e) for e in getattr(scenario, "events", [])] + [
        (float(m.position), "measure", m) for m in getattr(scenario, "measures", [])
    ]
    objects.sort(key=lambda item: item[0])

    for week, kind, obj in objects:
        delta_w = week - last_week
        baseline_running += float(scenario.baseline_trend) * delta_w
        strategy_running += float(scenario.baseline_trend) * delta_w

        if kind == "event":
            strategy_running += event_strategy_delta(obj, scenario)
            accuracy_running += event_accuracy_delta(obj)
        else:
            strategy_running += measure_strategy_delta(obj, scenario)
            accuracy_running += measure_accuracy_delta(obj)

        checkpoints.append(float(week))
        baseline_values.append(round(baseline_running, 3))
        strategy_values.append(round(strategy_running, 3))
        accuracy_values.append(max(0.0, min(100.0, round(accuracy_running, 2))))
        last_week = week

    max_week = max([float(getattr(p, "end_week", 52.0)) for p in getattr(scenario, "phases", [])] + [52.0])
    if checkpoints[-1] != max_week:
        delta_w = max_week - last_week
        baseline_running += float(scenario.baseline_trend) * delta_w
        strategy_running += float(scenario.baseline_trend) * delta_w
        checkpoints.append(max_week)
        baseline_values.append(round(baseline_running, 3))
        strategy_values.append(round(strategy_running, 3))
        accuracy_values.append(max(0.0, min(100.0, round(accuracy_running, 2))))

    return checkpoints, baseline_values, strategy_values, accuracy_values
