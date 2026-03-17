
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "scenario_data.json"


@dataclass
class Phase:
    id: str
    name: str
    start: float
    end: float
    description: str = ""


@dataclass
class TimelineEvent:
    id: str
    title: str
    phase_id: str
    position: float
    description: str
    impact_score: float
    direction: str
    risk_score: float = 0.0
    strategy_effect: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Measure:
    id: str
    title: str
    phase_id: str
    position: float
    description: str
    status: str
    effectiveness_score: float
    strategy_alignment: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Scenario:
    id: str
    name: str
    vision_goal: str
    baseline_accuracy: float
    target_value: float
    current_value: float
    baseline_trend: float = 0.0
    strategy_sensitivity: float = 1.0
    notes: str = ""
    phases: List[Phase] = field(default_factory=list)
    events: List[TimelineEvent] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def default_dataset() -> Dict[str, List[dict]]:
    scenario = Scenario(
        id=new_id("scenario"),
        name="Wasserversorgung Stadt Nord",
        vision_goal="Stabile Wasserversorgung unter unsicheren Bedingungen sichern",
        baseline_accuracy=72.0,
        target_value=100.0,
        current_value=86.0,
        baseline_trend=0.05,
        strategy_sensitivity=1.0,
        notes="Beispielszenario zur Demonstration des Tools.",
        phases=[
            Phase(id=new_id("phase"), name="Früherkennung", start=0, end=25, description="Monitoring, Signale und erste Abweichungen."),
            Phase(id=new_id("phase"), name="Belastung", start=25, end=55, description="Ereignisse erzeugen Druck auf die Strategie."),
            Phase(id=new_id("phase"), name="Stabilisierung", start=55, end=80, description="Gegenmaßnahmen werden geplant und aktiviert."),
            Phase(id=new_id("phase"), name="Resilienz", start=80, end=100, description="Nachsteuerung, Lernen und Absicherung."),
        ],
        events=[],
        measures=[],
    )
    scenario.events = [
        TimelineEvent(
            id=new_id("event"),
            title="Lange Trockenperiode",
            phase_id=scenario.phases[1].id,
            position=34,
            description="Sinkende Pegelstände erhöhen Versorgungsdruck.",
            impact_score=7.5,
            direction="negative",
            risk_score=8.0,
            strategy_effect=-6.0,
            metrics={"Pegel": -18, "Verbrauch": 11},
        ),
        TimelineEvent(
            id=new_id("event"),
            title="Anstieg Verbrauchsspitzen",
            phase_id=scenario.phases[1].id,
            position=44,
            description="Lastspitzen verschlechtern die operative Reserve.",
            impact_score=6.0,
            direction="negative",
            risk_score=6.5,
            strategy_effect=-4.5,
            metrics={"Tageslast": 13, "Spitzenbedarf": 9},
        ),
    ]
    scenario.measures = [
        Measure(
            id=new_id("measure"),
            title="Notfallbrunnen aktivieren",
            phase_id=scenario.phases[2].id,
            position=67,
            description="Zusätzliche Versorgungskapazität bereitstellen.",
            status="active",
            effectiveness_score=6.5,
            strategy_alignment=6.0,
            metrics={"Reservekapazität": 20},
        ),
        Measure(
            id=new_id("measure"),
            title="Verbrauchskommunikation an Bürger",
            phase_id=scenario.phases[1].id,
            position=46,
            description="Freiwillige Reduktion des Wasserverbrauchs.",
            status="planned",
            effectiveness_score=4.0,
            strategy_alignment=2.5,
            metrics={"erwartete Einsparung": 8},
        ),
    ]
    return {"scenarios": [asdict(scenario)]}


def load_data() -> Dict[str, List[dict]]:
    if not DATA_FILE.exists():
        data = default_dataset()
        save_data(data)
        return data
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: Dict[str, List[dict]]) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_scenarios() -> List[Scenario]:
    raw = load_data().get("scenarios", [])
    scenarios: List[Scenario] = []
    for item in raw:
        phases = [
            Phase(
                id=p.get("id", new_id("phase")),
                name=p.get("name", "Phase"),
                start=float(p.get("start", 0.0)),
                end=float(p.get("end", 100.0)),
                description=p.get("description", ""),
            )
            for p in item.get("phases", [])
        ]
        events = [
            TimelineEvent(
                id=e.get("id", new_id("event")),
                title=e.get("title", "Ereignis"),
                phase_id=e.get("phase_id", phases[0].id if phases else ""),
                position=float(e.get("position", 0.0)),
                description=e.get("description", ""),
                impact_score=float(e.get("impact_score", 0.0)),
                direction=e.get("direction", "neutral"),
                risk_score=float(e.get("risk_score", 0.0)),
                strategy_effect=float(e.get("strategy_effect", 0.0)),
                metrics=e.get("metrics", {}),
            )
            for e in item.get("events", [])
        ]
        measures = [
            Measure(
                id=m.get("id", new_id("measure")),
                title=m.get("title", "Maßnahme"),
                phase_id=m.get("phase_id", phases[0].id if phases else ""),
                position=float(m.get("position", 0.0)),
                description=m.get("description", ""),
                status=m.get("status", "planned"),
                effectiveness_score=float(m.get("effectiveness_score", 0.0)),
                strategy_alignment=float(m.get("strategy_alignment", 0.0)),
                metrics=m.get("metrics", {}),
            )
            for m in item.get("measures", [])
        ]
        scenarios.append(
            Scenario(
                id=item.get("id", new_id("scenario")),
                name=item.get("name", "Szenario"),
                vision_goal=item.get("vision_goal", ""),
                baseline_accuracy=float(item.get("baseline_accuracy", 50.0)),
                target_value=float(item.get("target_value", 100.0)),
                current_value=float(item.get("current_value", 0.0)),
                baseline_trend=float(item.get("baseline_trend", 0.0)),
                strategy_sensitivity=float(item.get("strategy_sensitivity", 1.0)),
                notes=item.get("notes", ""),
                phases=phases,
                events=events,
                measures=measures,
            )
        )
    return scenarios


def persist_scenarios(scenarios: List[Scenario]) -> None:
    save_data({"scenarios": [asdict(s) for s in scenarios]})


def find_scenario(scenarios: List[Scenario], scenario_id: str) -> Optional[Scenario]:
    return next((s for s in scenarios if s.id == scenario_id), None)


def phase_name_map(scenario: Scenario) -> Dict[str, str]:
    return {p.id: p.name for p in scenario.phases}


def calculate_deviation(scenario: Scenario) -> float:
    return scenario.target_value - scenario.current_value


def calculate_accuracy(scenario: Scenario) -> float:
    negative_pressure = sum(max(0.0, e.impact_score + 0.7 * e.risk_score) for e in scenario.events if e.direction == "negative")
    positive_support = sum(max(0.0, e.impact_score + max(0.0, e.strategy_effect)) for e in scenario.events if e.direction == "positive")
    measure_support = sum(
        max(0.0, m.effectiveness_score + 0.8 * m.strategy_alignment)
        for m in scenario.measures
        if m.status in {"active", "completed"}
    )
    score = scenario.baseline_accuracy - negative_pressure * 1.05 + positive_support * 0.7 + measure_support * 0.9
    return max(0.0, min(100.0, round(score, 1)))


def scenario_pressure(scenario: Scenario) -> float:
    return round(
        sum((e.impact_score + e.risk_score) for e in scenario.events if e.direction == "negative")
        - sum(m.effectiveness_score + m.strategy_alignment for m in scenario.measures if m.status in {"active", "completed"}),
        1,
    )


def wrap_text(text: str, width: int = 26) -> str:
    words = text.split()
    if not words:
        return text
    lines, line = [], []
    current = 0
    for word in words:
        add = len(word) + (1 if line else 0)
        if current + add <= width:
            line.append(word)
            current += add
        else:
            lines.append(" ".join(line))
            line = [word]
            current = len(word)
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)


def event_y(event: TimelineEvent, sensitivity: float = 1.0) -> float:
    intensity = abs(event.strategy_effect) * 0.18 + event.impact_score * 0.12 + event.risk_score * 0.10
    direction_boost = 0.38 if event.direction == "negative" else 0.14 if event.direction == "positive" else 0.05
    return 1.25 + min(3.0, (intensity + direction_boost) * max(0.75, sensitivity))


def measure_y(measure: Measure, sensitivity: float = 1.0) -> float:
    intensity = abs(measure.strategy_alignment) * 0.18 + measure.effectiveness_score * 0.12
    status_boost = 0.32 if measure.status == "active" else 0.18 if measure.status == "completed" else 0.08
    return -1.25 - min(3.0, (intensity + status_boost) * max(0.75, sensitivity))


def strategy_series(scenario: Scenario) -> tuple[list[float], list[float], list[float], list[float]]:
    checkpoints = [0.0]
    baseline_values = [scenario.current_value]
    strategy_values = [scenario.current_value]
    accuracy_values = [scenario.baseline_accuracy]

    baseline_running = scenario.current_value
    strategy_running = scenario.current_value
    accuracy_running = scenario.baseline_accuracy
    last_pos = 0.0

    items = sorted(
        [(e.position, "event", e) for e in scenario.events] + [(m.position, "measure", m) for m in scenario.measures],
        key=lambda x: x[0],
    )

    for pos, kind, obj in items:
        delta_t = max(0.0, pos - last_pos)
        baseline_running += scenario.baseline_trend * delta_t
        strategy_running += scenario.baseline_trend * delta_t

        if kind == "event":
            event = obj
            directional_factor = -1.0 if event.direction == "negative" else 0.6 if event.direction == "positive" else 0.0
            scenario_push = event.strategy_effect + directional_factor * event.impact_score + (
                -0.55 * event.risk_score if event.direction == "negative" else 0.15 * event.risk_score
            )
            strategy_running += scenario_push * scenario.strategy_sensitivity
            if event.direction == "negative":
                accuracy_running -= event.impact_score * 1.05 + event.risk_score * 0.85
            elif event.direction == "positive":
                accuracy_running += event.impact_score * 0.8 + max(0.0, event.strategy_effect) * 0.55
        else:
            measure = obj
            activation = 1.0 if measure.status in {"active", "completed"} else 0.45
            strategy_running += (
                measure.effectiveness_score * 0.45 + measure.strategy_alignment
            ) * activation * scenario.strategy_sensitivity
            accuracy_running += (
                measure.effectiveness_score * 0.9 + max(0.0, measure.strategy_alignment) * 0.55
            ) * activation

        checkpoints.append(float(pos))
        baseline_values.append(round(baseline_running, 3))
        strategy_values.append(round(strategy_running, 3))
        accuracy_values.append(max(0.0, min(100.0, round(accuracy_running, 2))))
        last_pos = pos

    if checkpoints[-1] != 100.0:
        delta_t = 100.0 - last_pos
        baseline_running += scenario.baseline_trend * delta_t
        strategy_running += scenario.baseline_trend * delta_t
        checkpoints.append(100.0)
        baseline_values.append(round(baseline_running, 3))
        strategy_values.append(round(strategy_running, 3))
        accuracy_values.append(max(0.0, min(100.0, round(accuracy_running, 2))))

    return checkpoints, baseline_values, strategy_values, accuracy_values


def value_to_y(value: float, min_val: float, max_val: float, low: float = -1.05, high: float = 1.05) -> float:
    if max_val - min_val < 1e-9:
        return (low + high) / 2
    ratio = (value - min_val) / (max_val - min_val)
    return low + ratio * (high - low)


def build_timeline_figure(scenario: Scenario) -> go.Figure:
    fig = go.Figure()
    checkpoints, baseline_values, strategy_values, accuracy_values = strategy_series(scenario)
    all_values = baseline_values + strategy_values + [scenario.target_value, scenario.current_value]
    min_val = min(all_values) - 5
    max_val = max(all_values) + 5

    phase_fill = ["rgba(99, 102, 241, 0.06)", "rgba(14, 165, 233, 0.06)"]
    for idx, phase in enumerate(sorted(scenario.phases, key=lambda p: p.start)):
        fig.add_vrect(
            x0=phase.start, x1=phase.end, annotation_text=phase.name, annotation_position="top left",
            annotation_font=dict(size=12, color="#334155"), fillcolor=phase_fill[idx % 2],
            opacity=1, line_width=0, layer="below",
        )
        if phase.description:
            fig.add_annotation(
                x=(phase.start + phase.end) / 2, y=3.75, text=wrap_text(phase.description, 30),
                showarrow=False, font=dict(size=10, color="#475569"),
                align="center", bgcolor="rgba(255,255,255,0.7)",
            )

    fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=0, line=dict(width=3, color="#475569"))
    target_y = value_to_y(scenario.target_value, min_val, max_val)
    fig.add_shape(type="line", x0=0, x1=100, y0=target_y, y1=target_y, line=dict(width=1.2, color="#94a3b8", dash="dash"))
    fig.add_annotation(x=100, y=target_y + 0.06, text=f"Zielwert {scenario.target_value:.1f}", showarrow=False, xanchor="right", font=dict(size=11, color="#475569"))

    baseline_y = [value_to_y(v, min_val, max_val) for v in baseline_values]
    strategy_y = [value_to_y(v, min_val, max_val) for v in strategy_values]
    accuracy_y = [-3.15 + (a / 100.0) * 0.9 for a in accuracy_values]

    fig.add_trace(go.Scatter(
        x=checkpoints, y=baseline_y, mode="lines+markers", name="Normale Trendlinie",
        line=dict(color="#94a3b8", width=3), marker=dict(size=6, color="#94a3b8"),
        customdata=[[round(v, 2)] for v in baseline_values],
        hovertemplate="Zeitpunkt: %{x}<br>Normaltrend: %{customdata[0]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=checkpoints, y=strategy_y, mode="lines+markers", name="Strategieverlauf",
        line=dict(color="#2563eb", width=5), marker=dict(size=8, color="#2563eb"),
        customdata=[[round(v, 2)] for v in strategy_values],
        hovertemplate="Zeitpunkt: %{x}<br>Strategieverlauf: %{customdata[0]}<extra></extra>",
    ))

    if scenario.events:
        fig.add_trace(go.Scatter(
            x=[e.position for e in scenario.events],
            y=[event_y(e, scenario.strategy_sensitivity) for e in scenario.events],
            mode="markers+text",
            text=[wrap_text(e.title) for e in scenario.events],
            textposition="top center",
            marker=dict(
                size=[16 + abs(e.strategy_effect) * 2.5 + e.risk_score * 1.3 for e in scenario.events],
                symbol="circle",
                color=[e.strategy_effect for e in scenario.events],
                colorscale="RdBu", cmin=-10, cmax=10,
                line=dict(width=1.2, color="#ffffff"),
            ),
            customdata=[[e.description, e.direction, e.impact_score, e.risk_score, e.strategy_effect,
                         ", ".join([f"{k}: {v}" for k, v in e.metrics.items()]) or "keine Kennzahlen"] for e in scenario.events],
            hovertemplate=(
                "<b>%{text}</b><br>Beschreibung: %{customdata[0]}<br>Richtung: %{customdata[1]}<br>"
                "Einflussstärke: %{customdata[2]}<br>Risikostärke: %{customdata[3]}<br>"
                "Direkter Effekt auf Strategie: %{customdata[4]}<br>Kennzahlen: %{customdata[5]}<extra></extra>"
            ),
            name="Szenarien",
        ))
        strategy_lookup = {round(x, 4): y for x, y in zip(checkpoints, strategy_y)}
        for e in scenario.events:
            top_y = event_y(e, scenario.strategy_sensitivity)
            anchor_y = strategy_lookup.get(round(float(e.position), 4), strategy_y[0])
            fig.add_shape(type="line", x0=e.position, x1=e.position, y0=0.02, y1=top_y - 0.14, line=dict(width=1, color="#64748b", dash="dot"))
            fig.add_shape(type="line", x0=e.position, x1=e.position, y0=min(top_y - 0.08, anchor_y), y1=max(top_y - 0.08, anchor_y), line=dict(width=1.6, color="#2563eb", dash="dash"))
            fig.add_annotation(x=e.position, y=(top_y + anchor_y) / 2, text=f"Δ {e.strategy_effect:+.1f}", showarrow=False, xshift=18, font=dict(size=11, color="#1d4ed8"), bgcolor="rgba(255,255,255,0.88)")

    if scenario.measures:
        fig.add_trace(go.Scatter(
            x=[m.position for m in scenario.measures],
            y=[measure_y(m, scenario.strategy_sensitivity) for m in scenario.measures],
            mode="markers+text",
            text=[wrap_text(m.title) for m in scenario.measures],
            textposition="bottom center",
            marker=dict(
                size=[16 + abs(m.strategy_alignment) * 2.1 + m.effectiveness_score * 1.4 for m in scenario.measures],
                symbol="diamond",
                color=[m.strategy_alignment for m in scenario.measures],
                colorscale="Blues", cmin=0, cmax=10,
                line=dict(width=1.2, color="#ffffff"),
            ),
            customdata=[[m.description, m.status, m.effectiveness_score, m.strategy_alignment,
                         ", ".join([f"{k}: {v}" for k, v in m.metrics.items()]) or "keine Kennzahlen"] for m in scenario.measures],
            hovertemplate=(
                "<b>%{text}</b><br>Beschreibung: %{customdata[0]}<br>Status: %{customdata[1]}<br>"
                "Wirksamkeit: %{customdata[2]}<br>Strategiebeitrag: %{customdata[3]}<br>"
                "Kennzahlen: %{customdata[4]}<extra></extra>"
            ),
            name="Gegenmaßnahmen",
        ))
        strategy_lookup = {round(x, 4): y for x, y in zip(checkpoints, strategy_y)}
        for m in scenario.measures:
            bottom_y = measure_y(m, scenario.strategy_sensitivity)
            anchor_y = strategy_lookup.get(round(float(m.position), 4), strategy_y[0])
            fig.add_shape(type="line", x0=m.position, x1=m.position, y0=-0.02, y1=bottom_y + 0.14, line=dict(width=1, color="#64748b", dash="dot"))
            fig.add_shape(type="line", x0=m.position, x1=m.position, y0=min(bottom_y + 0.08, anchor_y), y1=max(bottom_y + 0.08, anchor_y), line=dict(width=1.6, color="#0f766e", dash="dash"))
            fig.add_annotation(x=m.position, y=(bottom_y + anchor_y) / 2, text=f"Δ {m.strategy_alignment:+.1f}", showarrow=False, xshift=18, font=dict(size=11, color="#0f766e"), bgcolor="rgba(255,255,255,0.88)")

    fig.add_trace(go.Scatter(
        x=checkpoints, y=accuracy_y, mode="lines+markers", name="Szenarioplanungsgenauigkeit",
        line=dict(color="#7c3aed", width=3, dash="dot"), marker=dict(size=6, color="#7c3aed"),
        customdata=[[round(a, 1)] for a in accuracy_values],
        hovertemplate="Zeitpunkt: %{x}<br>Planungsgenauigkeit: %{customdata[0]} %<extra></extra>",
    ))

    fig.add_annotation(x=1, y=3.98, text="Oberhalb: Szenarien, Ereignisse und Risiken, die Druck auf die Strategie ausüben", showarrow=False, xanchor="left", font=dict(size=12, color="#334155"))
    fig.add_annotation(x=1, y=-4.08, text="Unterhalb: Gegenmaßnahmen, mit denen die Strategie reagieren und stabilisieren muss", showarrow=False, xanchor="left", font=dict(size=12, color="#334155"))

    fig.update_layout(
        height=860, paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        xaxis=dict(title="Zeit / Phasenfortschritt", range=[0, 100], showgrid=True, gridcolor="rgba(148,163,184,0.16)", zeroline=False),
        yaxis=dict(title="Auslenkung / Strategieverlauf", range=[-4.25, 4.25], showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False, showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(255,255,255,0.9)"),
        title=f"Strategieanalyse im Szenario: {scenario.name}",
        title_font=dict(size=22, color="#0f172a"),
        margin=dict(l=50, r=50, t=100, b=50),
        hoverlabel=dict(namelength=-1, bgcolor="#ffffff", font_color="#0f172a"),
    )
    return fig


def parse_metrics_input(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not text.strip():
        return metrics
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().replace(",", ".")
        if not key:
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    return metrics


def phase_position_bounds(scenario: Scenario, phase_id: str) -> tuple[float, float]:
    phase = next((p for p in scenario.phases if p.id == phase_id), None)
    if phase is None:
        return 0.0, 100.0
    return phase.start, phase.end


def main() -> None:
    st.set_page_config(page_title="Szenario-Management Tool", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background: #ffffff; color: #0f172a;}
        .block-container {padding-top: 1.0rem; padding-bottom: 1.0rem; max-width: 98rem;}
        div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0; padding: 0.6rem 0.8rem; border-radius: 0.8rem; box-shadow: 0 4px 14px rgba(15,23,42,0.04);}
        [data-testid="column"]:first-child > div {max-height: 83vh; overflow-y: auto; padding-right: 0.75rem;}
        [data-testid="column"]:nth-child(2) > div {position: sticky; top: 0.8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Szenario-Management Tool")
    st.caption("Die Darstellung analysiert die Strategie und zeigt, aus welchen Szenarien und Risiken Handlungsdruck entsteht und mit welchen Gegenmaßnahmen reagiert werden muss.")

    scenarios = get_scenarios()
    if not scenarios:
        persist_scenarios([])
        scenarios = get_scenarios()

    sidebar = st.sidebar
    sidebar.header("Szenarien")

    scenario_options = {s.name: s.id for s in scenarios}
    selected_name = sidebar.selectbox("Aktives Szenario", list(scenario_options.keys()))
    selected_id = scenario_options[selected_name]
    scenario = find_scenario(scenarios, selected_id)
    if scenario is None:
        st.error("Szenario konnte nicht geladen werden.")
        st.stop()

    with sidebar.expander("Neues Szenario anlegen"):
        with st.form("new_scenario_form", clear_on_submit=True):
            new_name = st.text_input("Name")
            new_goal = st.text_area("Visionsstrategisches Ziel")
            base_acc = st.slider("Basis-Planungsgenauigkeit", 0, 100, 70)
            target_val = st.number_input("Zielwert", value=100.0)
            current_val = st.number_input("Aktueller Wert", value=80.0)
            submitted = st.form_submit_button("Szenario anlegen")
            if submitted and new_name.strip():
                new_scenario = Scenario(
                    id=new_id("scenario"),
                    name=new_name.strip(),
                    vision_goal=new_goal.strip() or "Noch nicht definiert",
                    baseline_accuracy=float(base_acc),
                    target_value=float(target_val),
                    current_value=float(current_val),
                    baseline_trend=0.0,
                    strategy_sensitivity=1.0,
                    phases=[
                        Phase(id=new_id("phase"), name="Phase 1", start=0, end=25),
                        Phase(id=new_id("phase"), name="Phase 2", start=25, end=50),
                        Phase(id=new_id("phase"), name="Phase 3", start=50, end=75),
                        Phase(id=new_id("phase"), name="Phase 4", start=75, end=100),
                    ],
                )
                scenarios.append(new_scenario)
                persist_scenarios(scenarios)
                st.rerun()

    col_left, col_right = st.columns([1.05, 1.95], gap="large")

    with col_left:
        st.subheader("Szenario-Steuerung")

        with st.form("scenario_edit_form"):
            scenario.name = st.text_input("Szenarioname", value=scenario.name)
            scenario.vision_goal = st.text_area("Visionsstrategisches Ziel", value=scenario.vision_goal, height=100)
            c1, c2 = st.columns(2)
            with c1:
                scenario.baseline_accuracy = float(st.slider("Basis-Planungsgenauigkeit", 0, 100, int(scenario.baseline_accuracy)))
                scenario.current_value = float(st.number_input("Aktueller Wert", value=float(scenario.current_value)))
                scenario.baseline_trend = float(st.slider("Normale Trendlinie pro Zeiteinheit", -1.0, 1.0, float(scenario.baseline_trend), 0.05))
            with c2:
                scenario.target_value = float(st.number_input("Zielwert", value=float(scenario.target_value)))
                scenario.strategy_sensitivity = float(st.slider("Strategie-Sensitivität", 0.5, 2.5, float(scenario.strategy_sensitivity), 0.05))
                scenario.notes = st.text_area("Notizen", value=scenario.notes, height=80)
            if st.form_submit_button("Szenario speichern"):
                persist_scenarios(scenarios)
                st.success("Szenario gespeichert.")

        st.markdown("---")
        st.subheader("Phasen definieren")
        for idx, phase in enumerate(scenario.phases):
            with st.expander(f"{idx + 1}. {phase.name}"):
                phase.name = st.text_input(f"Phasenname {idx+1}", value=phase.name, key=f"phase_name_{phase.id}")
                p1, p2 = st.columns(2)
                with p1:
                    phase.start = float(st.number_input(f"Start {phase.name}", min_value=0.0, max_value=100.0, value=float(phase.start), key=f"phase_start_{phase.id}"))
                with p2:
                    phase.end = float(st.number_input(f"Ende {phase.name}", min_value=0.0, max_value=100.0, value=float(phase.end), key=f"phase_end_{phase.id}"))
                phase.description = st.text_area("Beschreibung", value=phase.description, key=f"phase_desc_{phase.id}")
        if st.button("Phasen speichern"):
            scenario.phases = sorted(scenario.phases, key=lambda p: p.start)
            persist_scenarios(scenarios)
            st.success("Phasen gespeichert.")

        with st.expander("Neue Phase hinzufügen"):
            with st.form("new_phase_form", clear_on_submit=True):
                phase_name = st.text_input("Neue Phase")
                c1, c2 = st.columns(2)
                with c1:
                    phase_start = st.number_input("Start", min_value=0.0, max_value=100.0, value=0.0)
                with c2:
                    phase_end = st.number_input("Ende", min_value=0.0, max_value=100.0, value=10.0)
                phase_desc = st.text_area("Beschreibung")
                if st.form_submit_button("Phase anlegen") and phase_name.strip():
                    scenario.phases.append(Phase(id=new_id("phase"), name=phase_name.strip(), start=float(phase_start), end=float(phase_end), description=phase_desc.strip()))
                    scenario.phases = sorted(scenario.phases, key=lambda p: p.start)
                    persist_scenarios(scenarios)
                    st.rerun()

        st.markdown("---")
        with st.expander("Ereignisse oberhalb des Zeitstrahls = Szenarien", expanded=True):
            with st.form("new_event_form", clear_on_submit=True):
                title = st.text_input("Titel des Szenarios")
                phase_id = st.selectbox("Phase", options=[p.id for p in scenario.phases], format_func=lambda x: phase_name_map(scenario).get(x, x))
                min_pos, max_pos = phase_position_bounds(scenario, phase_id)
                position = st.slider("Position auf dem Zeitstrahl", min_value=float(min_pos), max_value=float(max_pos), value=float(min_pos))
                description = st.text_area("Beschreibung")
                impact_score = st.slider("Einflussstärke", 0.0, 10.0, 5.0, 0.5)
                risk_score = st.slider("Risikostärke", 0.0, 10.0, 5.0, 0.5)
                strategy_effect = st.slider("Direkter Effekt auf die Strategie", -10.0, 10.0, 0.0, 0.5)
                direction = st.selectbox("Richtung", ["negative", "positive", "neutral"])
                metrics_text = st.text_area("Kennzahlen (eine pro Zeile: Name: Wert)")
                if st.form_submit_button("Szenario hinzufügen") and title.strip():
                    scenario.events.append(TimelineEvent(
                        id=new_id("event"), title=title.strip(), phase_id=phase_id, position=float(position),
                        description=description.strip(), impact_score=float(impact_score), direction=direction,
                        risk_score=float(risk_score), strategy_effect=float(strategy_effect),
                        metrics=parse_metrics_input(metrics_text),
                    ))
                    persist_scenarios(scenarios)
                    st.rerun()

        with st.expander("Gegenmaßnahmen unterhalb des Zeitstrahls = Gegenmaßnahmen", expanded=True):
            with st.form("new_measure_form", clear_on_submit=True):
                title = st.text_input("Titel der Gegenmaßnahme")
                phase_id = st.selectbox("Phase für Gegenmaßnahme", options=[p.id for p in scenario.phases], format_func=lambda x: phase_name_map(scenario).get(x, x))
                min_pos, max_pos = phase_position_bounds(scenario, phase_id)
                position = st.slider("Position", min_value=float(min_pos), max_value=float(max_pos), value=float(min_pos), key="measure_pos")
                description = st.text_area("Beschreibung", key="measure_desc")
                status = st.selectbox("Status", ["planned", "active", "completed"])
                effectiveness_score = st.slider("Wirksamkeit", 0.0, 10.0, 5.0, 0.5)
                strategy_alignment = st.slider("Beitrag zur Strategie", 0.0, 10.0, 0.0, 0.5)
                metrics_text = st.text_area("Kennzahlen (eine pro Zeile: Name: Wert)", key="measure_metrics")
                if st.form_submit_button("Gegenmaßnahme hinzufügen") and title.strip():
                    scenario.measures.append(Measure(
                        id=new_id("measure"), title=title.strip(), phase_id=phase_id, position=float(position),
                        description=description.strip(), status=status, effectiveness_score=float(effectiveness_score),
                        strategy_alignment=float(strategy_alignment), metrics=parse_metrics_input(metrics_text),
                    ))
                    persist_scenarios(scenarios)
                    st.rerun()

    with col_right:
        st.subheader("Analyse der Strategie")
        deviation = calculate_deviation(scenario)
        accuracy = calculate_accuracy(scenario)
        pressure = scenario_pressure(scenario)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Zielwert", f"{scenario.target_value:.1f}")
        m2.metric("Aktueller Wert", f"{scenario.current_value:.1f}")
        m3.metric("Auslenkung", f"{deviation:.1f}")
        m4.metric("Planungsgenauigkeit", f"{accuracy:.1f} %")
        m5.metric("Strategiedruck", f"{pressure:.1f}")

        st.plotly_chart(build_timeline_figure(scenario), width="stretch")

        st.markdown("### Analyseobjekte")
        tabs = st.tabs(["Szenarien", "Gegenmaßnahmen", "Datenexport"])
        with tabs[0]:
            if scenario.events:
                event_df = pd.DataFrame([{
                    "Szenario": e.title, "Phase": phase_name_map(scenario).get(e.phase_id, e.phase_id),
                    "Position": e.position, "Richtung": e.direction, "Einfluss": e.impact_score,
                    "Risiko": e.risk_score, "Strategieeffekt": e.strategy_effect,
                    "Kennzahlen": ", ".join([f"{k}: {v}" for k, v in e.metrics.items()]) or "—",
                } for e in sorted(scenario.events, key=lambda x: x.position)])
                st.dataframe(event_df, width="stretch")
            else:
                st.info("Noch keine Szenarien vorhanden.")
        with tabs[1]:
            if scenario.measures:
                measure_df = pd.DataFrame([{
                    "Gegenmaßnahme": m.title, "Phase": phase_name_map(scenario).get(m.phase_id, m.phase_id),
                    "Position": m.position, "Status": m.status, "Wirksamkeit": m.effectiveness_score,
                    "Strategiebeitrag": m.strategy_alignment,
                    "Kennzahlen": ", ".join([f"{k}: {v}" for k, v in m.metrics.items()]) or "—",
                } for m in sorted(scenario.measures, key=lambda x: x.position)])
                st.dataframe(measure_df, width="stretch")
            else:
                st.info("Noch keine Gegenmaßnahmen vorhanden.")
        with tabs[2]:
            export_payload = asdict(scenario)
            st.download_button("Aktives Szenario als JSON exportieren", data=json.dumps(export_payload, indent=2, ensure_ascii=False), file_name=f"{scenario.name.lower().replace(' ', '_')}.json", mime="application/json")
            uploaded = st.file_uploader("Szenario aus JSON importieren", type=["json"])
            if uploaded is not None:
                try:
                    payload = json.load(uploaded)
                    imported = Scenario(
                        id=payload.get("id", new_id("scenario")),
                        name=payload["name"],
                        vision_goal=payload.get("vision_goal", ""),
                        baseline_accuracy=float(payload.get("baseline_accuracy", 70)),
                        target_value=float(payload.get("target_value", 100)),
                        current_value=float(payload.get("current_value", 80)),
                        baseline_trend=float(payload.get("baseline_trend", 0.0)),
                        strategy_sensitivity=float(payload.get("strategy_sensitivity", 1.0)),
                        notes=payload.get("notes", ""),
                        phases=[Phase(id=p.get("id", new_id("phase")), name=p.get("name", "Phase"), start=float(p.get("start", 0.0)), end=float(p.get("end", 100.0)), description=p.get("description", "")) for p in payload.get("phases", [])],
                        events=[TimelineEvent(id=e.get("id", new_id("event")), title=e.get("title", "Ereignis"), phase_id=e.get("phase_id", ""), position=float(e.get("position", 0.0)), description=e.get("description", ""), impact_score=float(e.get("impact_score", 0.0)), direction=e.get("direction", "neutral"), risk_score=float(e.get("risk_score", 0.0)), strategy_effect=float(e.get("strategy_effect", 0.0)), metrics=e.get("metrics", {})) for e in payload.get("events", [])],
                        measures=[Measure(id=m.get("id", new_id("measure")), title=m.get("title", "Maßnahme"), phase_id=m.get("phase_id", ""), position=float(m.get("position", 0.0)), description=m.get("description", ""), status=m.get("status", "planned"), effectiveness_score=float(m.get("effectiveness_score", 0.0)), strategy_alignment=float(m.get("strategy_alignment", 0.0)), metrics=m.get("metrics", {})) for m in payload.get("measures", [])],
                    )
                    scenarios = [s for s in scenarios if s.id != imported.id]
                    scenarios.append(imported)
                    persist_scenarios(scenarios)
                    st.success("Szenario importiert. Bitte im Seitenmenü auswählen oder die Seite neu laden.")
                except Exception as ex:
                    st.error(f"Import fehlgeschlagen: {ex}")

    st.markdown("---")
    st.markdown("**Fachlogik des Prototyps**: Im Zentrum steht die Strategie. Oberhalb des Zeitstrahls werden Szenarien, Ereignisse und Risiken visualisiert, die auf die Strategie einwirken. Unterhalb werden Gegenmaßnahmen dargestellt, mit denen die Strategie reagieren, stabilisieren und gegensteuern muss.")


if __name__ == "__main__":
    main()
