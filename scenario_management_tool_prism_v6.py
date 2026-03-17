
from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "scenario_data.json"

STRATEGIC_LANES = {
    "strategie_1": {"label": "Strategieverlauf 1", "angle_deg": 90},
    "strategie_2": {"label": "Strategieverlauf 2", "angle_deg": 18},
    "strategie_3": {"label": "Strategieverlauf 3", "angle_deg": -54},
    "strategie_4": {"label": "Strategieverlauf 4", "angle_deg": -126},
    "strategie_5": {"label": "Strategieverlauf 5", "angle_deg": 162},
}
STATUS_FACTOR = {"planned": 0.45, "active": 1.0, "completed": 0.8}
EPSILON_RADIUS = 0.15


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
    lanes: List[str] = field(default_factory=lambda: ["strategie_1"])
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
    lanes: List[str] = field(default_factory=lambda: ["strategie_1"])
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Scenario:
    id: str
    name: str
    vision_goal: str
    baseline_accuracy: float
    target_value: float
    current_value: float
    outward_drift: float = 0.10
    strategy_sensitivity: float = 1.0
    notes: str = ""
    phases: List[Phase] = field(default_factory=list)
    events: List[TimelineEvent] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_lane_list(value) -> List[str]:
    if isinstance(value, list):
        vals = [str(v) for v in value if str(v) in STRATEGIC_LANES]
        return vals or ["strategie_1"]
    if isinstance(value, str) and value in STRATEGIC_LANES:
        return [value]
    return ["strategie_1"]


def default_dataset() -> Dict[str, List[dict]]:
    phases = [
        Phase(id=new_id("phase"), name="Früherkennung", start=0, end=20, description="Frühe Signale und Monitoring."),
        Phase(id=new_id("phase"), name="Belastung", start=20, end=50, description="Szenarien erhöhen die Auslenkung."),
        Phase(id=new_id("phase"), name="Reaktion", start=50, end=75, description="Maßnahmen ziehen Verläufe zurück Richtung Nullinie."),
        Phase(id=new_id("phase"), name="Resilienz", start=75, end=100, description="Absicherung und Stabilisierung."),
    ]
    scenario = Scenario(
        id=new_id("scenario"),
        name="Wasserversorgung Stadt Nord",
        vision_goal="Stabile Wasserversorgung unter unsicheren Bedingungen sichern",
        baseline_accuracy=72.0,
        target_value=100.0,
        current_value=86.0,
        outward_drift=0.08,
        strategy_sensitivity=1.0,
        notes="Alle Strategieverläufe starten auf der Nullinie und driften bei Szenarien nach außen.",
        phases=phases,
        events=[
            TimelineEvent(
                id=new_id("event"),
                title="Lange Trockenperiode",
                phase_id=phases[1].id,
                position=28,
                description="Sinkende Pegelstände erhöhen den Druck.",
                impact_score=7.5,
                direction="negative",
                risk_score=8.0,
                strategy_effect=-6.0,
                lanes=["strategie_1", "strategie_3"],
                metrics={"Pegel": -18, "Verbrauch": 11},
            ),
            TimelineEvent(
                id=new_id("event"),
                title="Netzstörung Pumpwerk",
                phase_id=phases[1].id,
                position=47,
                description="Technische Störung verschärft die Lage.",
                impact_score=6.2,
                direction="negative",
                risk_score=7.0,
                strategy_effect=-5.0,
                lanes=["strategie_2"],
                metrics={"Ausfallminuten": 90},
            ),
        ],
        measures=[
            Measure(
                id=new_id("measure"),
                title="Notfallbrunnen aktivieren",
                phase_id=phases[2].id,
                position=58,
                description="Reservekapazität wird zugeschaltet.",
                status="active",
                effectiveness_score=6.5,
                strategy_alignment=6.0,
                lanes=["strategie_1", "strategie_2"],
                metrics={"Reservekapazität": 20},
            ),
            Measure(
                id=new_id("measure"),
                title="Krisenkommunikation",
                phase_id=phases[2].id,
                position=72,
                description="Kommunikation reduziert Unsicherheit.",
                status="planned",
                effectiveness_score=4.8,
                strategy_alignment=4.5,
                lanes=["strategie_5"],
                metrics={"Reichweite": 12},
            ),
        ],
    )
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
                title=e.get("title", "Szenario"),
                phase_id=e.get("phase_id", phases[0].id if phases else ""),
                position=float(e.get("position", 0.0)),
                description=e.get("description", ""),
                impact_score=float(e.get("impact_score", 0.0)),
                direction=e.get("direction", "neutral"),
                risk_score=float(e.get("risk_score", 0.0)),
                strategy_effect=float(e.get("strategy_effect", 0.0)),
                lanes=ensure_lane_list(e.get("lanes", e.get("lane", ["strategie_1"]))),
                metrics=e.get("metrics", {}),
            )
            for e in item.get("events", [])
        ]
        measures = [
            Measure(
                id=m.get("id", new_id("measure")),
                title=m.get("title", "Gegenmaßnahme"),
                phase_id=m.get("phase_id", phases[0].id if phases else ""),
                position=float(m.get("position", 0.0)),
                description=m.get("description", ""),
                status=m.get("status", "planned"),
                effectiveness_score=float(m.get("effectiveness_score", 0.0)),
                strategy_alignment=float(m.get("strategy_alignment", 0.0)),
                lanes=ensure_lane_list(m.get("lanes", m.get("lane", ["strategie_1"]))),
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
                outward_drift=float(item.get("outward_drift", item.get("baseline_trend", 0.0))),
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
    negative_pressure = sum(
        max(0.0, e.impact_score + 0.7 * e.risk_score) * max(1, len(e.lanes)) * 0.6
        for e in scenario.events if e.direction == "negative"
    )
    measure_support = sum(
        max(0.0, m.effectiveness_score + 0.8 * m.strategy_alignment) * max(1, len(m.lanes)) * 0.5
        for m in scenario.measures if m.status in {"active", "completed"}
    )
    score = scenario.baseline_accuracy - negative_pressure * 1.05 + measure_support * 0.9
    return max(0.0, min(100.0, round(score, 1)))


def scenario_pressure(scenario: Scenario) -> float:
    return round(
        sum((e.impact_score + e.risk_score) * max(1, len(e.lanes)) * 0.6 for e in scenario.events if e.direction == "negative")
        - sum((m.effectiveness_score + m.strategy_alignment) * STATUS_FACTOR.get(m.status, 0.5) * max(1, len(m.lanes)) * 0.6 for m in scenario.measures),
        1,
    )


def phase_position_bounds(scenario: Scenario, phase_id: str) -> tuple[float, float]:
    phase = next((p for p in scenario.phases if p.id == phase_id), None)
    if phase is None:
        return 0.0, 100.0
    return phase.start, phase.end


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


def build_breakpoints(scenario: Scenario, lane: str):
    points = {0.0, 100.0}
    points.update(float(e.position) for e in scenario.events if lane in e.lanes)
    points.update(float(m.position) for m in scenario.measures if lane in m.lanes)
    xs = sorted(points)

    current_slope = scenario.outward_drift / 100.0
    value = 0.0
    last_x = 0.0
    ys = [0.0]

    events_by_pos = {}
    for e in scenario.events:
        if lane in e.lanes:
            events_by_pos.setdefault(float(e.position), []).append(e)

    measures_by_pos = {}
    for m in scenario.measures:
        if lane in m.lanes:
            measures_by_pos.setdefault(float(m.position), []).append(m)

    for x in xs[1:]:
        dx = x - last_x
        value = max(0.0, value + current_slope * dx)
        event_delta = 0.0
        for e in events_by_pos.get(float(x), []):
            lane_share = 1.0 / max(1, len(e.lanes))
            event_delta += (0.10 * e.impact_score + 0.08 * e.risk_score + 0.05 * abs(e.strategy_effect)) * lane_share
        measure_delta = 0.0
        for m in measures_by_pos.get(float(x), []):
            lane_share = 1.0 / max(1, len(m.lanes))
            measure_delta += (0.11 * m.effectiveness_score + 0.11 * m.strategy_alignment) * STATUS_FACTOR.get(m.status, 0.5) * lane_share
        current_slope = max(0.0, current_slope + event_delta - measure_delta)
        ys.append(value)
        last_x = x

    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def lane_signal(scenario: Scenario, lane: str, x: np.ndarray) -> np.ndarray:
    bx, by = build_breakpoints(scenario, lane)
    return np.interp(x, bx, by)


def total_effect_signal(scenario: Scenario, x: np.ndarray) -> np.ndarray:
    total = np.zeros_like(x, dtype=float)
    for lane in STRATEGIC_LANES:
        total += lane_signal(scenario, lane, x)
    return total / max(1, len(STRATEGIC_LANES))


def build_unwrapped_timeline_figure(scenario: Scenario, show_total: bool = True, selected_lanes: Optional[List[str]] = None) -> go.Figure:
    fig = go.Figure()
    x = np.linspace(0, 100, 500)
    selected_lanes = selected_lanes or list(STRATEGIC_LANES.keys())

    for i, phase in enumerate(sorted(scenario.phases, key=lambda p: p.start)):
        fig.add_vrect(
            x0=phase.start,
            x1=phase.end,
            fillcolor=["rgba(59,130,246,0.05)", "rgba(16,185,129,0.05)"][i % 2],
            line_width=0,
            layer="below",
            annotation_text=phase.name,
            annotation_position="top left",
        )

    fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=0, line=dict(width=3, color="#334155"))
    fig.add_annotation(x=100, y=0.08, text="Nullinie / Timeline", showarrow=False, xanchor="right", font=dict(size=11, color="#334155"))

    if show_total:
        total = total_effect_signal(scenario, x)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=total,
                mode="lines",
                name="Gesamtwirkung",
                line=dict(width=5, dash="dash"),
                hovertemplate="Gesamtwirkung<br>Zeit: %{x:.1f}<br>Auslenkung: %{y:.3f}<extra></extra>",
            )
        )

    for lane in selected_lanes:
        signal = lane_signal(scenario, lane, x)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=signal,
                mode="lines",
                name=STRATEGIC_LANES[lane]["label"],
                line=dict(width=3),
                hovertemplate=f"{STRATEGIC_LANES[lane]['label']}<br>Zeit: %{{x:.1f}}<br>Auslenkung: %{{y:.3f}}<extra></extra>",
            )
        )

    for e in scenario.events:
        for lane in [l for l in e.lanes if l in selected_lanes]:
            local_y = lane_signal(scenario, lane, np.array([e.position]))[0]
            fig.add_trace(
                go.Scatter(
                    x=[e.position],
                    y=[local_y],
                    mode="markers+text",
                    text=[e.title],
                    textposition="top center",
                    marker=dict(size=8 + e.risk_score, symbol="circle"),
                    showlegend=False,
                    hovertemplate=(f"<b>{e.title}</b><br>{STRATEGIC_LANES[lane]['label']}<br>Einfluss: {e.impact_score:.1f}<br>Risiko: {e.risk_score:.1f}<br>Strategieeffekt: {e.strategy_effect:.1f}<extra></extra>"),
                )
            )

    for m in scenario.measures:
        for lane in [l for l in m.lanes if l in selected_lanes]:
            local_y = lane_signal(scenario, lane, np.array([m.position]))[0]
            fig.add_trace(
                go.Scatter(
                    x=[m.position],
                    y=[local_y],
                    mode="markers+text",
                    text=[m.title],
                    textposition="bottom center",
                    marker=dict(size=8 + m.effectiveness_score, symbol="diamond"),
                    showlegend=False,
                    hovertemplate=(f"<b>{m.title}</b><br>{STRATEGIC_LANES[lane]['label']}<br>Wirksamkeit: {m.effectiveness_score:.1f}<br>Strategiebeitrag: {m.strategy_alignment:.1f}<br>Status: {m.status}<extra></extra>"),
                )
            )

    fig.update_layout(
        height=650,
        title=f"Ausgerollte Fläche des strategischen Körpers: {scenario.name}",
        xaxis=dict(title="Zeit / Phasenverlauf", range=[0, 104]),
        yaxis=dict(title="Auslenkung relativ zur Nullinie"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        margin=dict(l=30, r=60, t=60, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def build_prism_figure(
    scenario: Scenario,
    highlighted_lanes: Optional[List[str]] = None,
) -> go.Figure:
    z = np.linspace(0, 100, 220)
    lane_keys = list(STRATEGIC_LANES.keys())
    highlighted_lanes = highlighted_lanes or lane_keys
    angles = np.array([math.radians(STRATEGIC_LANES[k]["angle_deg"]) for k in lane_keys])

    profiles = {k: EPSILON_RADIUS + 0.55 * lane_signal(scenario, k, z) for k in lane_keys}

    n = len(lane_keys)
    X = np.zeros((n + 1, len(z)))
    Y = np.zeros((n + 1, len(z)))
    Z = np.zeros((n + 1, len(z)))
    for i, lane in enumerate(lane_keys):
        r = profiles[lane]
        X[i, :] = r * np.cos(angles[i])
        Y[i, :] = r * np.sin(angles[i])
        Z[i, :] = z
    X[n, :] = X[0, :]
    Y[n, :] = Y[0, :]
    Z[n, :] = Z[0, :]

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.tile(np.arange(n + 1).reshape(-1, 1), (1, len(z))), colorscale="Blues", showscale=False, opacity=0.62, hoverinfo="skip"))

    for i, lane in enumerate(lane_keys):
        is_highlight = lane in highlighted_lanes
        fig.add_trace(
            go.Scatter3d(
                x=X[i, :], y=Y[i, :], z=Z[i, :],
                mode="lines",
                line=dict(width=10 if is_highlight else 5),
                opacity=1.0 if is_highlight else 0.25,
                name=STRATEGIC_LANES[lane]["label"],
                hovertemplate=f"{STRATEGIC_LANES[lane]['label']}<br>Zeit: %{{x:.1f}}<br>Auslenkung: %{{y:.3f}}, %{{z:.3f}}<extra></extra>",
            )
        )

    for e in scenario.events:
        for lane in e.lanes:
            i = lane_keys.index(lane)
            r = EPSILON_RADIUS + 0.55 * lane_signal(scenario, lane, np.array([e.position]))[0]
            angle = angles[i]
            x = (r + 0.03) * math.cos(angle)
            y = (r + 0.03) * math.sin(angle)
            z0 = e.position
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z0], mode="markers+text",
                marker=dict(size=6 + e.risk_score, symbol="circle"),
                text=[e.title if lane in highlighted_lanes else ""],
                textposition="top center",
                showlegend=False,
                opacity=1.0 if lane in highlighted_lanes else 0.18,
                hovertemplate=(f"<b>{e.title}</b><br>{STRATEGIC_LANES[lane]['label']}<br>Einfluss: {e.impact_score:.1f}<br>Risiko: {e.risk_score:.1f}<br>Strategieeffekt: {e.strategy_effect:.1f}<extra></extra>")
            ))

    for m in scenario.measures:
        for lane in m.lanes:
            i = lane_keys.index(lane)
            r = EPSILON_RADIUS + 0.55 * lane_signal(scenario, lane, np.array([m.position]))[0]
            angle = angles[i]
            x = max(EPSILON_RADIUS * 0.6, r - 0.02) * math.cos(angle)
            y = max(EPSILON_RADIUS * 0.6, r - 0.02) * math.sin(angle)
            z0 = m.position
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z0], mode="markers+text",
                marker=dict(size=6 + m.effectiveness_score, symbol="diamond"),
                text=[m.title if lane in highlighted_lanes else ""],
                textposition="bottom center",
                showlegend=False,
                opacity=1.0 if lane in highlighted_lanes else 0.18,
                hovertemplate=(f"<b>{m.title}</b><br>{STRATEGIC_LANES[lane]['label']}<br>Wirksamkeit: {m.effectiveness_score:.1f}<br>Strategiebeitrag: {m.strategy_alignment:.1f}<br>Status: {m.status}<extra></extra>")
            ))

    fig.update_layout(
        height=720,
        title=f"Strategischer Körper in 3D: {scenario.name}",
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=2.8, y=0.8, z=0.8),
            dragmode=False,
            camera=dict(
                eye=dict(
                    x=0.02,
                    y=2.35 * math.cos(math.radians(rotation_deg)),
                    z=2.35 * math.sin(math.radians(rotation_deg)),
                ),
                up=dict(x=1, y=0, z=0),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        legend=dict(orientation="h"),
    )
    return fig


def build_cross_section_figure(scenario: Scenario, section_week: float, selected_lanes: Optional[List[str]] = None) -> go.Figure:
    fig = go.Figure()
    lane_keys = selected_lanes or list(STRATEGIC_LANES.keys())
    angles = [math.radians(STRATEGIC_LANES[k]["angle_deg"]) for k in lane_keys]
    radii = [EPSILON_RADIUS + 0.55 * lane_signal(scenario, k, np.array([section_week]))[0] for k in lane_keys]

    xs = [r * math.cos(a) for r, a in zip(radii, angles)]
    ys = [r * math.sin(a) for r, a in zip(radii, angles)]
    xs_closed = xs + [xs[0]]
    ys_closed = ys + [ys[0]]

    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=6), name="Nullzentrum"))
    fig.add_trace(go.Scatter(x=xs_closed, y=ys_closed, mode="lines+markers", fill="toself", line=dict(width=3), marker=dict(size=9), name="Strategischer Querschnitt"))

    for x, y, lane, r in zip(xs, ys, lane_keys, radii):
        fig.add_annotation(x=x, y=y, text=f"{STRATEGIC_LANES[lane]['label']}<br>{r-EPSILON_RADIUS:+.2f}", showarrow=False, font=dict(size=11))

    fig.update_layout(
        title=f"Querschnitt des strategischen Körpers bei Zeitpunkt {section_week:.1f}",
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Szenario-Management Tool", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background: #ffffff; color: #0f172a;}
        .block-container {padding-top: 1.0rem; padding-bottom: 1.0rem; max-width: 98rem;}
        div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0; padding: 0.6rem 0.8rem; border-radius: 0.8rem; box-shadow: 0 4px 14px rgba(15,23,42,0.04);}
        [data-testid="column"]:first-child > div {max-height: 84vh; overflow-y: auto; padding-right: 0.75rem;}
        [data-testid="column"]:nth-child(2) > div {position: sticky; top: 0.75rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Szenario-Management Tool")
    st.caption("Alle Strategieverläufe starten auf der Nullinie. Szenarien treiben sie linear weiter nach außen, Maßnahmen reduzieren die Steigung zurück Richtung Nullinie.")

    scenarios = get_scenarios()
    if not scenarios:
        save_data(default_dataset())
        scenarios = get_scenarios()

    if "save_status" not in st.session_state:
        st.session_state.save_status = ""

    sidebar = st.sidebar
    sidebar.header("Szenarien")
    options = {s.name: s.id for s in scenarios}
    selected_name = sidebar.selectbox("Aktives Szenario", list(options.keys()))
    scenario = find_scenario(scenarios, options[selected_name])
    if scenario is None:
        st.error("Szenario konnte nicht geladen werden.")
        st.stop()

    if sidebar.button("Alle eingetragenen Werte speichern", width="stretch"):
        persist_scenarios(scenarios)
        st.session_state.save_status = "scenario_data.json wurde erfolgreich überschrieben."
    if st.session_state.save_status:
        sidebar.success(st.session_state.save_status)

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
                    outward_drift=0.0,
                    strategy_sensitivity=1.0,
                    phases=[
                        Phase(id=new_id("phase"), name="Phase 1", start=0, end=20),
                        Phase(id=new_id("phase"), name="Phase 2", start=20, end=50),
                        Phase(id=new_id("phase"), name="Phase 3", start=50, end=75),
                        Phase(id=new_id("phase"), name="Phase 4", start=75, end=100),
                    ],
                )
                scenarios.append(new_scenario)
                persist_scenarios(scenarios)
                st.session_state.save_status = "Neues Szenario gespeichert."
                st.rerun()

    left, right = st.columns([1.05, 1.95], gap="large")

    with left:
        st.subheader("Szenario-Steuerung")
        with st.form("scenario_edit_form"):
            scenario.name = st.text_input("Szenarioname", value=scenario.name)
            scenario.vision_goal = st.text_area("Visionsstrategisches Ziel", value=scenario.vision_goal, height=100)
            c1, c2 = st.columns(2)
            with c1:
                scenario.baseline_accuracy = float(st.slider("Basis-Planungsgenauigkeit", 0, 100, int(scenario.baseline_accuracy)))
                scenario.current_value = float(st.number_input("Aktueller Wert", value=float(scenario.current_value)))
                scenario.outward_drift = float(st.slider("Normale Auswärtsdrift", 0.0, 2.0, float(scenario.outward_drift), 0.05))
            with c2:
                scenario.target_value = float(st.number_input("Zielwert", value=float(scenario.target_value)))
                scenario.strategy_sensitivity = float(st.slider("Strategie-Sensitivität", 0.5, 2.5, float(scenario.strategy_sensitivity), 0.05))
                scenario.notes = st.text_area("Notizen", value=scenario.notes, height=80)
            if st.form_submit_button("Szenario speichern"):
                persist_scenarios(scenarios)
                st.session_state.save_status = "Szenario gespeichert."
                st.success("Szenario gespeichert und JSON überschrieben.")

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
        if st.button("Phasen speichern", width="stretch"):
            scenario.phases = sorted(scenario.phases, key=lambda p: p.start)
            persist_scenarios(scenarios)
            st.session_state.save_status = "Phasen gespeichert."
            st.success("Phasen gespeichert und JSON überschrieben.")

        with st.expander("Szenarien", expanded=True):
            with st.form("new_event_form", clear_on_submit=True):
                title = st.text_input("Titel des Szenarios")
                phase_id = st.selectbox("Phase", options=[p.id for p in scenario.phases], format_func=lambda x: phase_name_map(scenario).get(x, x))
                min_pos, max_pos = phase_position_bounds(scenario, phase_id)
                position = st.slider("Position auf dem Verlauf", min_value=float(min_pos), max_value=float(max_pos), value=float(min_pos))
                lanes = st.multiselect("Strategische Verläufe auswählen", options=list(STRATEGIC_LANES.keys()), default=["strategie_1"], format_func=lambda x: STRATEGIC_LANES[x]["label"])
                description = st.text_area("Beschreibung")
                impact_score = st.slider("Einflussstärke", 0.0, 10.0, 5.0, 0.5)
                risk_score = st.slider("Risikostärke", 0.0, 10.0, 5.0, 0.5)
                strategy_effect = st.slider("Direkter Effekt auf Strategie", -10.0, 10.0, 0.0, 0.5)
                direction = st.selectbox("Richtung", ["negative", "positive", "neutral"])
                metrics_text = st.text_area("Kennzahlen (eine pro Zeile: Name: Wert)")
                if st.form_submit_button("Szenario hinzufügen") and title.strip():
                    scenario.events.append(TimelineEvent(
                        id=new_id("event"), title=title.strip(), phase_id=phase_id, position=float(position),
                        description=description.strip(), impact_score=float(impact_score), direction=direction,
                        risk_score=float(risk_score), strategy_effect=float(strategy_effect),
                        lanes=lanes or ["strategie_1"], metrics=parse_metrics_input(metrics_text),
                    ))
                    persist_scenarios(scenarios)
                    st.session_state.save_status = "Szenario hinzugefügt."
                    st.rerun()

        with st.expander("Gegenmaßnahmen unterhalb des Zeitstrahls = Gegenmaßnahmen", expanded=True):
            with st.form("new_measure_form", clear_on_submit=True):
                title = st.text_input("Titel der Gegenmaßnahme")
                phase_id = st.selectbox("Phase für Gegenmaßnahme", options=[p.id for p in scenario.phases], format_func=lambda x: phase_name_map(scenario).get(x, x))
                min_pos, max_pos = phase_position_bounds(scenario, phase_id)
                position = st.slider("Position", min_value=float(min_pos), max_value=float(max_pos), value=float(min_pos), key="measure_pos")
                lanes = st.multiselect("Strategische Verläufe für Maßnahme", options=list(STRATEGIC_LANES.keys()), default=["strategie_1"], format_func=lambda x: STRATEGIC_LANES[x]["label"], key="measure_lanes")
                description = st.text_area("Beschreibung", key="measure_desc")
                status = st.selectbox("Status", ["planned", "active", "completed"])
                effectiveness_score = st.slider("Wirksamkeit", 0.0, 10.0, 5.0, 0.5)
                strategy_alignment = st.slider("Strategiebeitrag", 0.0, 10.0, 0.0, 0.5)
                metrics_text = st.text_area("Kennzahlen (eine pro Zeile: Name: Wert)", key="measure_metrics")
                if st.form_submit_button("Gegenmaßnahme hinzufügen") and title.strip():
                    scenario.measures.append(Measure(
                        id=new_id("measure"), title=title.strip(), phase_id=phase_id, position=float(position),
                        description=description.strip(), status=status, effectiveness_score=float(effectiveness_score),
                        strategy_alignment=float(strategy_alignment), lanes=lanes or ["strategie_1"],
                        metrics=parse_metrics_input(metrics_text),
                    ))
                    persist_scenarios(scenarios)
                    st.session_state.save_status = "Gegenmaßnahme hinzugefügt."
                    st.rerun()

        if st.button("Jetzt alles in JSON speichern", width="stretch"):
            persist_scenarios(scenarios)
            st.session_state.save_status = "Alle eingetragenen Werte wurden in scenario_data.json gespeichert."
            st.success(st.session_state.save_status)

    with right:
        st.subheader("Analyse des strategischen Körpers")
        deviation = calculate_deviation(scenario)
        accuracy = calculate_accuracy(scenario)
        pressure = scenario_pressure(scenario)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Zielwert", f"{scenario.target_value:.1f}")
        m2.metric("Aktueller Wert", f"{scenario.current_value:.1f}")
        m3.metric("Auslenkung", f"{deviation:.1f}")
        m4.metric("Planungsgenauigkeit", f"{accuracy:.1f} %")
        m5.metric("Strategiedruck", f"{pressure:.1f}")

        highlighted_lanes = st.multiselect("Im 3D-Körper hervorheben", options=list(STRATEGIC_LANES.keys()), default=list(STRATEGIC_LANES.keys()), format_func=lambda x: STRATEGIC_LANES[x]["label"])
        unwrapped_lanes = st.multiselect("In der ausgerollten Fläche anzeigen", options=list(STRATEGIC_LANES.keys()), default=list(STRATEGIC_LANES.keys()), format_func=lambda x: STRATEGIC_LANES[x]["label"])
        show_total = st.toggle("Gesamtwirkung in ausgerollter Fläche anzeigen", value=True)
        section_week = st.slider("Querschnitt / Zeitpunkt", 0.0, 100.0, 50.0, 1.0)

        t1, t2, t3 = st.tabs(["3D-Körper", "Ausgerollte Fläche", "Querschnitt"])
        with t1:
            st.caption("Die Zeitachse steht vertikal. Die Maus kann den strategischen Körper interaktiv um diese Achse rotieren.")
            st.plotly_chart(
                build_prism_figure(scenario, highlighted_lanes),
                width="stretch",
                config={"scrollZoom": False, "displayModeBar": False},
            )
        with t2:
            st.plotly_chart(build_unwrapped_timeline_figure(scenario, show_total=show_total, selected_lanes=unwrapped_lanes), width="stretch")
        with t3:
            st.plotly_chart(build_cross_section_figure(scenario, section_week, selected_lanes=highlighted_lanes or list(STRATEGIC_LANES.keys())), width="stretch")

        st.markdown("### Analyseobjekte")
        tabs2 = st.tabs(["Szenarien", "Gegenmaßnahmen"])
        with tabs2[0]:
            if scenario.events:
                st.dataframe(pd.DataFrame([{
                    "Szenario": e.title,
                    "Phase": phase_name_map(scenario).get(e.phase_id, e.phase_id),
                    "Verläufe": ", ".join(STRATEGIC_LANES[l]["label"] for l in e.lanes),
                    "Position": e.position,
                    "Einfluss": e.impact_score,
                    "Risiko": e.risk_score,
                    "Strategieeffekt": e.strategy_effect,
                } for e in sorted(scenario.events, key=lambda v: v.position)]), width="stretch")
            else:
                st.info("Noch keine Szenarien vorhanden.")
        with tabs2[1]:
            if scenario.measures:
                st.dataframe(pd.DataFrame([{
                    "Gegenmaßnahme": m.title,
                    "Phase": phase_name_map(scenario).get(m.phase_id, m.phase_id),
                    "Verläufe": ", ".join(STRATEGIC_LANES[l]["label"] for l in m.lanes),
                    "Position": m.position,
                    "Wirksamkeit": m.effectiveness_score,
                    "Strategiebeitrag": m.strategy_alignment,
                    "Status": m.status,
                } for m in sorted(scenario.measures, key=lambda v: v.position)]), width="stretch")
            else:
                st.info("Noch keine Gegenmaßnahmen vorhanden.")

    st.markdown("---")
    st.markdown(
        "**Fachlogik**: Alle Strategieverläufe starten auf der Nullinie. Szenarien erhöhen ab ihrem Zeitpunkt linear die weitere Auswärtsbewegung. "
        "Maßnahmen reduzieren ab ihrem Zeitpunkt die weitere Steigung Richtung Nullinie. "
        "Der Speichern-Button überschreibt `scenario_data.json` und zeigt den Status an. Die 3D-Ansicht verwendet eine vertikale Zeitachse und kann interaktiv per Maus um diese Achse gedreht werden."
    )


if __name__ == "__main__":
    main()
