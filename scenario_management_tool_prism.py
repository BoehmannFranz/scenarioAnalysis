
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

# -------------------------------------------------
# Data model
# -------------------------------------------------
STRATEGIC_LANES = {
    "strategie_1": {"label": "Strategieverlauf 1", "angle_deg": 90},
    "strategie_2": {"label": "Strategieverlauf 2", "angle_deg": 18},
    "strategie_3": {"label": "Strategieverlauf 3", "angle_deg": -54},
    "strategie_4": {"label": "Strategieverlauf 4", "angle_deg": -126},
    "strategie_5": {"label": "Strategieverlauf 5", "angle_deg": 162},
}

STATUS_FACTOR = {"planned": 0.55, "active": 1.0, "completed": 0.85}


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
    lane: str = "strategie_1"
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
    lane: str = "strategie_1"
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
    phases = [
        Phase(id=new_id("phase"), name="Früherkennung", start=0, end=20, description="Frühe Signale und Monitoring."),
        Phase(id=new_id("phase"), name="Belastung", start=20, end=50, description="Szenarien erzeugen Druck auf die Strategie."),
        Phase(id=new_id("phase"), name="Reaktion", start=50, end=75, description="Gegenmaßnahmen stabilisieren die Strategie."),
        Phase(id=new_id("phase"), name="Resilienz", start=75, end=100, description="Lernen, Verstetigen und Absichern."),
    ]
    scenario = Scenario(
        id=new_id("scenario"),
        name="Wasserversorgung Stadt Nord",
        vision_goal="Stabile Wasserversorgung unter unsicheren Bedingungen sichern",
        baseline_accuracy=72.0,
        target_value=100.0,
        current_value=86.0,
        baseline_trend=0.06,
        strategy_sensitivity=1.0,
        notes="Visualisierung als strategischer Zylinderkörper mit polygonalem Querschnitt.",
        phases=phases,
        events=[
            TimelineEvent(
                id=new_id("event"),
                title="Lange Trockenperiode",
                phase_id=phases[1].id,
                position=28,
                description="Sinkende Pegelstände erhöhen Versorgungsdruck.",
                impact_score=7.5,
                direction="negative",
                risk_score=8.0,
                strategy_effect=-6.0,
                lane="strategie_1",
                metrics={"Pegel": -18, "Verbrauch": 11},
            ),
            TimelineEvent(
                id=new_id("event"),
                title="Netzstörung Pumpwerk",
                phase_id=phases[1].id,
                position=47,
                description="Technische Störung im Betrieb verschärft die Lage.",
                impact_score=6.2,
                direction="negative",
                risk_score=7.0,
                strategy_effect=-5.0,
                lane="strategie_2",
                metrics={"Ausfallminuten": 90},
            ),
            TimelineEvent(
                id=new_id("event"),
                title="Politischer Erwartungsdruck",
                phase_id=phases[2].id,
                position=63,
                description="Kommunikative Anforderungen erhöhen den Steuerungsdruck.",
                impact_score=4.5,
                direction="negative",
                risk_score=5.2,
                strategy_effect=-3.5,
                lane="strategie_5",
                metrics={"Medienanfragen": 14},
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
                lane="strategie_1",
                metrics={"Reservekapazität": 20},
            ),
            Measure(
                id=new_id("measure"),
                title="Betriebsumstellung",
                phase_id=phases[2].id,
                position=68,
                description="Systemlast wird aktiv umverteilt.",
                status="active",
                effectiveness_score=5.5,
                strategy_alignment=5.0,
                lane="strategie_2",
                metrics={"Lastreduktion": 9},
            ),
            Measure(
                id=new_id("measure"),
                title="Krisenkommunikation",
                phase_id=phases[2].id,
                position=72,
                description="Transparente Kommunikation reduziert Unsicherheit.",
                status="planned",
                effectiveness_score=4.8,
                strategy_alignment=4.5,
                lane="strategie_5",
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
                lane=e.get("lane", "strategie_1"),
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
                lane=m.get("lane", "strategie_1"),
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
        - sum((m.effectiveness_score + m.strategy_alignment) * STATUS_FACTOR.get(m.status, 0.5) for m in scenario.measures),
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


def lane_signal(scenario: Scenario, lane: str, x: np.ndarray) -> np.ndarray:
    y = scenario.baseline_trend * (x / 8.0)
    for e in scenario.events:
        if e.lane != lane:
            continue
        width = 6.0 + 0.35 * e.risk_score
        sign = -1.0 if e.direction == "negative" else 1.0 if e.direction == "positive" else 0.0
        amplitude = sign * (0.65 * e.impact_score + 0.55 * e.risk_score + 0.8 * abs(e.strategy_effect))
        y += amplitude * np.exp(-((x - e.position) ** 2) / (2 * width ** 2))
    for m in scenario.measures:
        if m.lane != lane:
            continue
        width = 7.0
        amplitude = (0.75 * m.effectiveness_score + 0.9 * m.strategy_alignment) * STATUS_FACTOR.get(m.status, 0.5)
        y += amplitude * np.exp(-((x - m.position) ** 2) / (2 * width ** 2))
    return y * scenario.strategy_sensitivity


def build_unwrapped_timeline_figure(scenario: Scenario) -> go.Figure:
    fig = go.Figure()
    x = np.linspace(0, 100, 300)

    phase_colors = ["rgba(59,130,246,0.05)", "rgba(16,185,129,0.05)"]
    for i, phase in enumerate(sorted(scenario.phases, key=lambda p: p.start)):
        fig.add_vrect(
            x0=phase.start,
            x1=phase.end,
            fillcolor=phase_colors[i % 2],
            line_width=0,
            layer="below",
            annotation_text=phase.name,
            annotation_position="top left",
        )

    lane_offsets = {}
    labels = list(STRATEGIC_LANES.keys())
    top = 3.0
    step = 1.45
    for idx, lane in enumerate(labels):
        lane_offsets[lane] = top - idx * step

    # baseline center line per lane
    for lane, meta in STRATEGIC_LANES.items():
        offset = lane_offsets[lane]
        signal = lane_signal(scenario, lane, x)
        y = offset + 0.18 * signal
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=meta["label"],
                line=dict(width=3),
                hovertemplate=f"{meta['label']}<br>Zeit: %{{x:.1f}}<br>Verlauf: %{{y:.2f}}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=101,
            y=offset,
            text=meta["label"],
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color="#334155"),
        )

    # scenario markers
    for e in scenario.events:
        offset = lane_offsets[e.lane]
        local_y = offset + 0.18 * lane_signal(scenario, e.lane, np.array([e.position]))[0]
        fig.add_trace(
            go.Scatter(
                x=[e.position],
                y=[local_y],
                mode="markers+text",
                text=[e.title],
                textposition="top center",
                marker=dict(size=8 + e.risk_score, symbol="circle"),
                name=f"Szenario {e.title}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{e.title}</b><br>{STRATEGIC_LANES[e.lane]['label']}<br>"
                    f"Einfluss: {e.impact_score:.1f}<br>Risiko: {e.risk_score:.1f}<br>"
                    f"Strategieeffekt: {e.strategy_effect:.1f}<extra></extra>"
                ),
            )
        )
        fig.add_shape(
            type="line",
            x0=e.position,
            x1=e.position,
            y0=offset,
            y1=local_y,
            line=dict(width=1, dash="dot"),
        )

    # measure markers
    for m in scenario.measures:
        offset = lane_offsets[m.lane]
        local_y = offset + 0.18 * lane_signal(scenario, m.lane, np.array([m.position]))[0]
        fig.add_trace(
            go.Scatter(
                x=[m.position],
                y=[local_y],
                mode="markers+text",
                text=[m.title],
                textposition="bottom center",
                marker=dict(size=8 + m.effectiveness_score, symbol="diamond"),
                name=f"Gegenmaßnahme {m.title}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{m.title}</b><br>{STRATEGIC_LANES[m.lane]['label']}<br>"
                    f"Wirksamkeit: {m.effectiveness_score:.1f}<br>Strategiebeitrag: {m.strategy_alignment:.1f}<br>"
                    f"Status: {m.status}<extra></extra>"
                ),
            )
        )
        fig.add_shape(
            type="line",
            x0=m.position,
            x1=m.position,
            y0=offset,
            y1=local_y,
            line=dict(width=1, dash="dot"),
        )

    fig.update_layout(
        height=600,
        title=f"Ausgerollter Strategiezylinder: {scenario.name}",
        xaxis=dict(title="Zeit / Phasenverlauf", range=[0, 104]),
        yaxis=dict(title="Strategische Verläufe", showticklabels=False),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        margin=dict(l=30, r=60, t=60, b=30),
        legend=dict(orientation="h"),
    )
    return fig


def build_prism_figure(scenario: Scenario) -> go.Figure:
    z = np.linspace(0, 100, 120)
    n = len(STRATEGIC_LANES)
    lane_keys = list(STRATEGIC_LANES.keys())
    angles = np.array([math.radians(STRATEGIC_LANES[k]["angle_deg"]) for k in lane_keys])

    profiles = {k: 6.0 + 0.22 * lane_signal(scenario, k, z) for k in lane_keys}

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
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=np.tile(np.arange(n + 1).reshape(-1, 1), (1, len(z))),
            colorscale="Blues",
            showscale=False,
            opacity=0.8,
            hoverinfo="skip",
            name="Strategischer Körper",
        )
    )

    for i, lane in enumerate(lane_keys):
        fig.add_trace(
            go.Scatter3d(
                x=X[i, :],
                y=Y[i, :],
                z=Z[i, :],
                mode="lines",
                line=dict(width=8),
                name=STRATEGIC_LANES[lane]["label"],
                hovertemplate=f"{STRATEGIC_LANES[lane]['label']}<br>Zeit: %{{z:.1f}}<extra></extra>",
            )
        )

    # events on prism
    for e in scenario.events:
        lane = e.lane
        i = lane_keys.index(lane)
        r = 6.0 + 0.22 * lane_signal(scenario, lane, np.array([e.position]))[0]
        angle = angles[i]
        x = (r + 0.75) * math.cos(angle)
        y = (r + 0.75) * math.sin(angle)
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[e.position],
                mode="markers+text",
                marker=dict(size=6 + e.risk_score, symbol="circle"),
                text=[e.title],
                textposition="top center",
                showlegend=False,
                hovertemplate=(
                    f"<b>{e.title}</b><br>{STRATEGIC_LANES[lane]['label']}<br>"
                    f"Einfluss: {e.impact_score:.1f}<br>Risiko: {e.risk_score:.1f}<br>"
                    f"Strategieeffekt: {e.strategy_effect:.1f}<extra></extra>"
                ),
            )
        )

    for m in scenario.measures:
        lane = m.lane
        i = lane_keys.index(lane)
        r = 6.0 + 0.22 * lane_signal(scenario, lane, np.array([m.position]))[0]
        angle = angles[i]
        x = (r - 0.6) * math.cos(angle)
        y = (r - 0.6) * math.sin(angle)
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[m.position],
                mode="markers+text",
                marker=dict(size=6 + m.effectiveness_score, symbol="diamond"),
                text=[m.title],
                textposition="bottom center",
                showlegend=False,
                hovertemplate=(
                    f"<b>{m.title}</b><br>{STRATEGIC_LANES[lane]['label']}<br>"
                    f"Wirksamkeit: {m.effectiveness_score:.1f}<br>Strategiebeitrag: {m.strategy_alignment:.1f}<br>"
                    f"Status: {m.status}<extra></extra>"
                ),
            )
        )

    for phase in scenario.phases:
        for phase_pos in [phase.start, phase.end]:
            xs, ys = [], []
            for i, lane in enumerate(lane_keys):
                r = profiles[lane][np.argmin(np.abs(z - phase_pos))]
                xs.append(r * math.cos(angles[i]))
                ys.append(r * math.sin(angles[i]))
            xs.append(xs[0]); ys.append(ys[0])
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=[phase_pos] * len(xs),
                    mode="lines",
                    line=dict(width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        height=700,
        title=f"Strategischer Körper in 3D: {scenario.name}",
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Zeit / Phasenverlauf", range=[0, 100]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.9),
            camera=dict(eye=dict(x=1.55, y=1.5, z=0.95)),
        ),
        legend=dict(orientation="h"),
    )
    return fig


def build_cross_section_figure(scenario: Scenario, section_week: float) -> go.Figure:
    fig = go.Figure()
    lane_keys = list(STRATEGIC_LANES.keys())
    angles = [math.radians(STRATEGIC_LANES[k]["angle_deg"]) for k in lane_keys]
    radii = [6.0 + 0.22 * lane_signal(scenario, k, np.array([section_week]))[0] for k in lane_keys]

    xs = [r * math.cos(a) for r, a in zip(radii, angles)]
    ys = [r * math.sin(a) for r, a in zip(radii, angles)]
    xs_closed = xs + [xs[0]]
    ys_closed = ys + [ys[0]]

    base_x = [6.0 * math.cos(a) for a in angles] + [6.0 * math.cos(angles[0])]
    base_y = [6.0 * math.sin(a) for a in angles] + [6.0 * math.sin(angles[0])]

    fig.add_trace(
        go.Scatter(
            x=base_x,
            y=base_y,
            mode="lines",
            line=dict(width=2, dash="dash"),
            name="Referenzquerschnitt",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs_closed,
            y=ys_closed,
            mode="lines+markers",
            fill="toself",
            line=dict(width=3),
            marker=dict(size=9),
            name="Strategischer Querschnitt",
        )
    )

    for x, y, lane, r in zip(xs, ys, lane_keys, radii):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"{STRATEGIC_LANES[lane]['label']}<br>{r-6.0:+.2f}",
            showarrow=False,
            font=dict(size=11),
        )

    fig.update_layout(
        title=f"Querschnitt des strategischen Körpers bei Zeitpunkt {section_week:.1f}",
        height=480,
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
    st.caption("Linke Seite Steuerung, rechte Seite strategischer Körper: 3D-Ansicht, ausgerollte Zylinderfläche und Querschnitt.")

    scenarios = get_scenarios()
    if not scenarios:
        save_data(default_dataset())
        scenarios = get_scenarios()

    sidebar = st.sidebar
    sidebar.header("Szenarien")
    options = {s.name: s.id for s in scenarios}
    selected_name = sidebar.selectbox("Aktives Szenario", list(options.keys()))
    scenario = find_scenario(scenarios, options[selected_name])
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
                        Phase(id=new_id("phase"), name="Phase 1", start=0, end=20),
                        Phase(id=new_id("phase"), name="Phase 2", start=20, end=50),
                        Phase(id=new_id("phase"), name="Phase 3", start=50, end=75),
                        Phase(id=new_id("phase"), name="Phase 4", start=75, end=100),
                    ],
                )
                scenarios.append(new_scenario)
                persist_scenarios(scenarios)
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
                scenario.baseline_trend = float(st.slider("Normale Drift", -1.0, 1.0, float(scenario.baseline_trend), 0.05))
            with c2:
                scenario.target_value = float(st.number_input("Zielwert", value=float(scenario.target_value)))
                scenario.strategy_sensitivity = float(st.slider("Strategie-Sensitivität", 0.5, 2.5, float(scenario.strategy_sensitivity), 0.05))
                scenario.notes = st.text_area("Notizen", value=scenario.notes, height=80)
            if st.form_submit_button("Szenario speichern"):
                persist_scenarios(scenarios)
                st.success("Szenario gespeichert.")

        st.markdown("---")
        st.subheader("Strategische Verläufe")
        st.dataframe(
            pd.DataFrame(
                [{"Key": k, "Verlauf": v["label"], "Ecke": i + 1} for i, (k, v) in enumerate(STRATEGIC_LANES.items())]
            ),
            width="stretch",
        )

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

        with st.expander("Ereignisse oberhalb des Zeitstrahls = Szenarien", expanded=True):
            with st.form("new_event_form", clear_on_submit=True):
                title = st.text_input("Titel des Szenarios")
                phase_id = st.selectbox("Phase", options=[p.id for p in scenario.phases], format_func=lambda x: phase_name_map(scenario).get(x, x))
                min_pos, max_pos = phase_position_bounds(scenario, phase_id)
                position = st.slider("Position auf dem Verlauf", min_value=float(min_pos), max_value=float(max_pos), value=float(min_pos))
                lane = st.selectbox("Strategischer Verlauf / Ecke", options=list(STRATEGIC_LANES.keys()), format_func=lambda x: STRATEGIC_LANES[x]["label"])
                description = st.text_area("Beschreibung")
                impact_score = st.slider("Einflussstärke", 0.0, 10.0, 5.0, 0.5)
                risk_score = st.slider("Risikostärke", 0.0, 10.0, 5.0, 0.5)
                strategy_effect = st.slider("Direkter Effekt auf Strategie", -10.0, 10.0, 0.0, 0.5)
                direction = st.selectbox("Richtung", ["negative", "positive", "neutral"])
                metrics_text = st.text_area("Kennzahlen (eine pro Zeile: Name: Wert)")
                if st.form_submit_button("Szenario hinzufügen") and title.strip():
                    scenario.events.append(
                        TimelineEvent(
                            id=new_id("event"),
                            title=title.strip(),
                            phase_id=phase_id,
                            position=float(position),
                            description=description.strip(),
                            impact_score=float(impact_score),
                            direction=direction,
                            risk_score=float(risk_score),
                            strategy_effect=float(strategy_effect),
                            lane=lane,
                            metrics=parse_metrics_input(metrics_text),
                        )
                    )
                    persist_scenarios(scenarios)
                    st.rerun()

        with st.expander("Gegenmaßnahmen unterhalb des Zeitstrahls = Gegenmaßnahmen", expanded=True):
            with st.form("new_measure_form", clear_on_submit=True):
                title = st.text_input("Titel der Gegenmaßnahme")
                phase_id = st.selectbox("Phase für Gegenmaßnahme", options=[p.id for p in scenario.phases], format_func=lambda x: phase_name_map(scenario).get(x, x))
                min_pos, max_pos = phase_position_bounds(scenario, phase_id)
                position = st.slider("Position", min_value=float(min_pos), max_value=float(max_pos), value=float(min_pos), key="measure_pos")
                lane = st.selectbox("Strategischer Verlauf / Ecke für Maßnahme", options=list(STRATEGIC_LANES.keys()), format_func=lambda x: STRATEGIC_LANES[x]["label"])
                description = st.text_area("Beschreibung", key="measure_desc")
                status = st.selectbox("Status", ["planned", "active", "completed"])
                effectiveness_score = st.slider("Wirksamkeit", 0.0, 10.0, 5.0, 0.5)
                strategy_alignment = st.slider("Strategiebeitrag", 0.0, 10.0, 0.0, 0.5)
                metrics_text = st.text_area("Kennzahlen (eine pro Zeile: Name: Wert)", key="measure_metrics")
                if st.form_submit_button("Gegenmaßnahme hinzufügen") and title.strip():
                    scenario.measures.append(
                        Measure(
                            id=new_id("measure"),
                            title=title.strip(),
                            phase_id=phase_id,
                            position=float(position),
                            description=description.strip(),
                            status=status,
                            effectiveness_score=float(effectiveness_score),
                            strategy_alignment=float(strategy_alignment),
                            lane=lane,
                            metrics=parse_metrics_input(metrics_text),
                        )
                    )
                    persist_scenarios(scenarios)
                    st.rerun()

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

        section_week = st.slider("Querschnitt / Zeitpunkt", 0.0, 100.0, 50.0, 1.0)
        t1, t2, t3 = st.tabs(["3D-Körper", "Ausgerollte Fläche", "Querschnitt"])
        with t1:
            st.plotly_chart(build_prism_figure(scenario), width="stretch")
        with t2:
            st.plotly_chart(build_unwrapped_timeline_figure(scenario), width="stretch")
        with t3:
            st.plotly_chart(build_cross_section_figure(scenario, section_week), width="stretch")

        st.markdown("### Analyseobjekte")
        tabs2 = st.tabs(["Szenarien", "Gegenmaßnahmen"])
        with tabs2[0]:
            if scenario.events:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Szenario": e.title,
                                "Phase": phase_name_map(scenario).get(e.phase_id, e.phase_id),
                                "Ecke / Verlauf": STRATEGIC_LANES[e.lane]["label"],
                                "Position": e.position,
                                "Einfluss": e.impact_score,
                                "Risiko": e.risk_score,
                                "Strategieeffekt": e.strategy_effect,
                            }
                            for e in sorted(scenario.events, key=lambda v: v.position)
                        ]
                    ),
                    width="stretch",
                )
            else:
                st.info("Noch keine Szenarien vorhanden.")
        with tabs2[1]:
            if scenario.measures:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Gegenmaßnahme": m.title,
                                "Phase": phase_name_map(scenario).get(m.phase_id, m.phase_id),
                                "Ecke / Verlauf": STRATEGIC_LANES[m.lane]["label"],
                                "Position": m.position,
                                "Wirksamkeit": m.effectiveness_score,
                                "Strategiebeitrag": m.strategy_alignment,
                                "Status": m.status,
                            }
                            for m in sorted(scenario.measures, key=lambda v: v.position)
                        ]
                    ),
                    width="stretch",
                )
            else:
                st.info("Noch keine Gegenmaßnahmen vorhanden.")

        st.caption("Die Visualisierung folgt jetzt der Skizzenlogik: mehrere strategische Verläufe über die Zeit, daraus ein räumlicher Körper, darunter der polygonale Querschnitt.")

    st.markdown("---")
    st.markdown(
        "**Fachlogik**: Jede Ecke des Körpers repräsentiert einen strategischen Verlauf. "
        "Szenarien drücken einzelne Verläufe aus dem Soll heraus. Gegenmaßnahmen stabilisieren diese Verläufe wieder. "
        "Der 3D-Körper zeigt die zeitliche Verdichtung, die ausgerollte Fläche zeigt die einzelnen Verlaufsbahnen und der Querschnitt zeigt die Form des Systems zu einem konkreten Zeitpunkt."
    )


if __name__ == "__main__":
    main()
