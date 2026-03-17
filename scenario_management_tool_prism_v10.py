
from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "scenario_data.json"

LANES = {
    "v1": {"label": "Strategischer Verlauf 1", "angle_deg": 90},
    "v2": {"label": "Strategischer Verlauf 2", "angle_deg": -30},
    "v3": {"label": "Strategischer Verlauf 3", "angle_deg": -150},
    "v4": {"label": "Strategischer Verlauf 4", "angle_deg": 210},
}
STATUS_FACTOR = {"planned": 0.45, "active": 1.0, "completed": 0.8}


@dataclass
class Phase:
    id: str
    name: str
    start: float
    end: float
    description: str = ""


@dataclass
class ScenarioEvent:
    id: str
    title: str
    phase_id: str
    position: float
    description: str
    intensity: float
    risk: float
    lanes: List[str] = field(default_factory=lambda: ["v1"])
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Measure:
    id: str
    title: str
    phase_id: str
    position: float
    description: str
    effectiveness: float
    status: str
    lanes: List[str] = field(default_factory=lambda: ["v1"])
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScenarioModel:
    id: str
    name: str
    vision_goal: str
    outward_drift: float = 0.08
    sensitivity: float = 1.0
    phases: List[Phase] = field(default_factory=list)
    events: List[ScenarioEvent] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_lanes(value) -> List[str]:
    if isinstance(value, list):
        vals = [str(v) for v in value if str(v) in LANES]
        return vals or ["v1"]
    if isinstance(value, str) and value in LANES:
        return [value]
    return ["v1"]


def default_model() -> ScenarioModel:
    phases = [
        Phase(new_id("phase"), "Früherkennung", 0, 20, "Frühe Signale und Beobachtung"),
        Phase(new_id("phase"), "Belastung", 20, 50, "Szenarien erhöhen die Auslenkung"),
        Phase(new_id("phase"), "Reaktion", 50, 75, "Maßnahmen ziehen Verläufe zurück"),
        Phase(new_id("phase"), "Stabilisierung", 75, 100, "Rückführung und Absicherung"),
    ]
    return ScenarioModel(
        id=new_id("scenario"),
        name="Szenario Wasserversorgung",
        vision_goal="Strategische Stabilität entlang der Zeitachse sichern",
        outward_drift=0.08,
        sensitivity=1.0,
        phases=phases,
        events=[
            ScenarioEvent(
                id=new_id("event"),
                title="Trockenperiode",
                phase_id=phases[1].id,
                position=28,
                description="Versorgungsdruck steigt.",
                intensity=7.5,
                risk=8.0,
                lanes=["v1", "v2"],
                metrics={"Pegel": -18},
            ),
            ScenarioEvent(
                id=new_id("event"),
                title="Technische Störung",
                phase_id=phases[1].id,
                position=46,
                description="Betriebsstörung verstärkt die Auslenkung.",
                intensity=6.0,
                risk=7.0,
                lanes=["v3"],
                metrics={"Ausfallminuten": 90},
            ),
        ],
        measures=[
            Measure(
                id=new_id("measure"),
                title="Reserve aktivieren",
                phase_id=phases[2].id,
                position=58,
                description="Zieht den Verlauf wieder an die Zeitachse heran.",
                effectiveness=6.5,
                status="active",
                lanes=["v1", "v2"],
                metrics={"Reserve": 20},
            ),
            Measure(
                id=new_id("measure"),
                title="Kommunikation",
                phase_id=phases[2].id,
                position=70,
                description="Stabilisiert Erwartungsdruck.",
                effectiveness=4.8,
                status="planned",
                lanes=["v4"],
                metrics={"Reichweite": 12},
            ),
        ],
    )


def save_model(model: ScenarioModel) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(asdict(model), f, indent=2, ensure_ascii=False)


def load_model() -> ScenarioModel:
    if not DATA_FILE.exists():
        model = default_model()
        save_model(model)
        return model
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    phases = [
        Phase(
            id=p.get("id", new_id("phase")),
            name=p.get("name", "Phase"),
            start=float(p.get("start", 0.0)),
            end=float(p.get("end", 100.0)),
            description=p.get("description", ""),
        )
        for p in raw.get("phases", [])
    ]
    return ScenarioModel(
        id=raw.get("id", new_id("scenario")),
        name=raw.get("name", "Szenario"),
        vision_goal=raw.get("vision_goal", ""),
        outward_drift=float(raw.get("outward_drift", 0.08)),
        sensitivity=float(raw.get("sensitivity", 1.0)),
        phases=phases,
        events=[
            ScenarioEvent(
                id=e.get("id", new_id("event")),
                title=e.get("title", "Szenario"),
                phase_id=e.get("phase_id", phases[0].id if phases else ""),
                position=float(e.get("position", 0.0)),
                description=e.get("description", ""),
                intensity=float(e.get("intensity", e.get("impact_score", 0.0))),
                risk=float(e.get("risk", e.get("risk_score", 0.0))),
                lanes=ensure_lanes(e.get("lanes", e.get("lane", ["v1"]))),
                metrics=e.get("metrics", {}),
            )
            for e in raw.get("events", [])
        ],
        measures=[
            Measure(
                id=m.get("id", new_id("measure")),
                title=m.get("title", "Maßnahme"),
                phase_id=m.get("phase_id", phases[0].id if phases else ""),
                position=float(m.get("position", 0.0)),
                description=m.get("description", ""),
                effectiveness=float(m.get("effectiveness", m.get("effectiveness_score", 0.0))),
                status=m.get("status", "planned"),
                lanes=ensure_lanes(m.get("lanes", m.get("lane", ["v1"]))),
                metrics=m.get("metrics", {}),
            )
            for m in raw.get("measures", [])
        ],
    )


def phase_name_map(model: ScenarioModel) -> Dict[str, str]:
    return {p.id: p.name for p in model.phases}


def phase_bounds(model: ScenarioModel, phase_id: str) -> tuple[float, float]:
    phase = next((p for p in model.phases if p.id == phase_id), None)
    if phase is None:
        return 0.0, 100.0
    return phase.start, phase.end


def parse_metrics(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip().replace(",", ".")
        if not k:
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def build_linear_profile(model: ScenarioModel, lane: str):
    # piecewise-linear slope model
    points = {0.0, 100.0}
    points.update(float(e.position) for e in model.events if lane in e.lanes)
    points.update(float(m.position) for m in model.measures if lane in m.lanes)
    xs = sorted(points)

    slope = model.outward_drift / 100.0
    value = 0.0
    last_x = 0.0
    ys = [0.0]

    events_by_x: Dict[float, List[ScenarioEvent]] = {}
    for e in model.events:
        if lane in e.lanes:
            events_by_x.setdefault(float(e.position), []).append(e)

    measures_by_x: Dict[float, List[Measure]] = {}
    for m in model.measures:
        if lane in m.lanes:
            measures_by_x.setdefault(float(m.position), []).append(m)

    for x in xs[1:]:
        dx = x - last_x
        value = max(0.0, value + slope * dx)
        for e in events_by_x.get(float(x), []):
            share = 1.0 / max(1, len(e.lanes))
            slope += (0.11 * e.intensity + 0.09 * e.risk) * share / 100.0
        for m in measures_by_x.get(float(x), []):
            share = 1.0 / max(1, len(m.lanes))
            slope -= (0.12 * m.effectiveness) * STATUS_FACTOR.get(m.status, 0.5) * share / 100.0
        slope = max(0.0, slope)
        ys.append(value * model.sensitivity)
        last_x = x

    return np.array(xs), np.array(ys)


def lane_radius(model: ScenarioModel, lane: str, z: np.ndarray) -> np.ndarray:
    bx, by = build_linear_profile(model, lane)
    return np.interp(z, bx, by)


def build_3d_axis_figure(model: ScenarioModel) -> go.Figure:
    z = np.linspace(0, 100, 220)
    lane_keys = list(LANES.keys())
    angles = {k: math.radians(v["angle_deg"]) for k, v in LANES.items()}

    fig = go.Figure()

    # central time axis
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[0, 100],
            mode="lines",
            line=dict(width=7, color="black"),
            name="Zeitachse",
            hoverinfo="skip",
        )
    )

    # phase rings + labels
    ring_t = np.linspace(0, 2 * np.pi, 80)
    for p in sorted(model.phases, key=lambda x: x.start):
        for z0 in [p.start, p.end]:
            r = 0.2
            fig.add_trace(
                go.Scatter3d(
                    x=r * np.cos(ring_t),
                    y=r * np.sin(ring_t),
                    z=np.full_like(ring_t, z0),
                    mode="lines",
                    line=dict(width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[0.0],
                y=[0.0],
                z=[(p.start + p.end) / 2.0],
                mode="text",
                text=[p.name],
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # strategic lanes: v1 visible in top/front, others around axis
    for lane in lane_keys:
        r = lane_radius(model, lane, z)
        angle = angles[lane]
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(width=9),
                name=LANES[lane]["label"],
                hovertemplate=f"{LANES[lane]['label']}<br>Zeit: %{{z:.1f}}<br>Auslenkung: %{{x:.3f}}, %{{y:.3f}}<extra></extra>",
            )
        )

    # event markers
    for e in model.events:
        for lane in e.lanes:
            angle = angles[lane]
            r = float(lane_radius(model, lane, np.array([e.position]))[0])
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            fig.add_trace(
                go.Scatter3d(
                    x=[x], y=[y], z=[e.position],
                    mode="markers+text",
                    marker=dict(size=6 + e.risk, symbol="circle"),
                    text=[e.title if lane == "v1" else ""],
                    textposition="top center",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{e.title}</b><br>{LANES[lane]['label']}<br>"
                        f"Intensität: {e.intensity:.1f}<br>Risiko: {e.risk:.1f}<extra></extra>"
                    ),
                )
            )

    # measure markers
    for m in model.measures:
        for lane in m.lanes:
            angle = angles[lane]
            r = float(lane_radius(model, lane, np.array([m.position]))[0])
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            fig.add_trace(
                go.Scatter3d(
                    x=[x], y=[y], z=[m.position],
                    mode="markers+text",
                    marker=dict(size=6 + m.effectiveness, symbol="diamond"),
                    text=[m.title if lane == "v1" else ""],
                    textposition="bottom center",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{m.title}</b><br>{LANES[lane]['label']}<br>"
                        f"Wirksamkeit: {m.effectiveness:.1f}<br>Status: {m.status}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=980,
        title=f"3D-Strategievisualisierung: {model.name}",
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=2.3),
            dragmode="orbit",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.1),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        legend=dict(orientation="h"),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="3D Strategievisualisierung", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background: #ffffff; color: #0f172a;}
        .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 112rem;}
        div[data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e2e8f0;
            padding: 0.6rem 0.8rem; border-radius: 0.8rem;
            box-shadow: 0 4px 14px rgba(15,23,42,0.04);
        }
        [data-testid="column"]:first-child > div {
            max-height: 84vh; overflow-y: auto; padding-right: 0.75rem;
        }
        [data-testid="column"]:nth-child(2) > div {
            position: sticky; top: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()

    st.title("3D Strategievisualisierung")
    st.caption(
        "In der Mitte liegt die Zeitachse, an Phasen orientiert. Alle strategischen Verläufe starten auf 0. "
        "Szenarien drücken die Linien von der Achse weg, Maßnahmen ziehen sie wieder an die Achse heran."
    )

    left, right = st.columns([0.9, 2.1], gap="large")

    with left:
        st.subheader("Szenario-Steuerung")

        with st.form("scenario_form"):
            model.name = st.text_input("Szenarioname", value=model.name)
            model.vision_goal = st.text_area("Visionsziel", value=model.vision_goal, height=90)
            c1, c2 = st.columns(2)
            with c1:
                model.outward_drift = float(st.slider("Grundauswärtsdrift", 0.0, 2.0, float(model.outward_drift), 0.05))
            with c2:
                model.sensitivity = float(st.slider("Sensitivität", 0.5, 3.0, float(model.sensitivity), 0.05))
            if st.form_submit_button("Modell speichern"):
                save_model(model)
                st.success("scenario_data.json überschrieben.")

        st.markdown("---")
        st.subheader("Phasen")
        for i, p in enumerate(model.phases):
            with st.expander(f"{i+1}. {p.name}"):
                p.name = st.text_input(f"Name {i+1}", value=p.name, key=f"pn_{p.id}")
                c1, c2 = st.columns(2)
                with c1:
                    p.start = float(st.number_input("Start", 0.0, 100.0, float(p.start), key=f"ps_{p.id}"))
                with c2:
                    p.end = float(st.number_input("Ende", 0.0, 100.0, float(p.end), key=f"pe_{p.id}"))
                p.description = st.text_area("Beschreibung", value=p.description, key=f"pd_{p.id}")
        if st.button("Phasen in JSON speichern", width="stretch"):
            model.phases = sorted(model.phases, key=lambda p: p.start)
            save_model(model)
            st.success("Phasen gespeichert.")

        st.markdown("---")
        with st.expander("Szenarien", expanded=True):
            with st.form("event_form", clear_on_submit=True):
                title = st.text_input("Titel des Szenarios")
                phase_id = st.selectbox("Phase", [p.id for p in model.phases], format_func=lambda x: phase_name_map(model).get(x, x))
                lo, hi = phase_bounds(model, phase_id)
                position = st.slider("Zeitpunkt", float(lo), float(hi), float(lo))
                lanes = st.multiselect("Betroffene Strategieverläufe", list(LANES.keys()), default=["v1"], format_func=lambda x: LANES[x]["label"])
                description = st.text_area("Beschreibung")
                intensity = st.slider("Intensität", 0.0, 10.0, 5.0, 0.5)
                risk = st.slider("Risikobewertung", 0.0, 10.0, 5.0, 0.5)
                metrics_text = st.text_area("Kennzahlen (Name: Wert)")
                if st.form_submit_button("Szenario hinzufügen") and title.strip():
                    model.events.append(
                        ScenarioEvent(
                            id=new_id("event"),
                            title=title.strip(),
                            phase_id=phase_id,
                            position=float(position),
                            description=description.strip(),
                            intensity=float(intensity),
                            risk=float(risk),
                            lanes=lanes or ["v1"],
                            metrics=parse_metrics(metrics_text),
                        )
                    )
                    save_model(model)
                    st.rerun()

        with st.expander("Gegenmaßnahmen", expanded=True):
            with st.form("measure_form", clear_on_submit=True):
                title = st.text_input("Titel der Maßnahme")
                phase_id = st.selectbox("Phase für Maßnahme", [p.id for p in model.phases], format_func=lambda x: phase_name_map(model).get(x, x))
                lo, hi = phase_bounds(model, phase_id)
                position = st.slider("Zeitpunkt der Maßnahme", float(lo), float(hi), float(lo), key="mpos")
                lanes = st.multiselect("Zieht diese Verläufe zurück", list(LANES.keys()), default=["v1"], format_func=lambda x: LANES[x]["label"], key="mlanes")
                description = st.text_area("Beschreibung", key="mdesc")
                effectiveness = st.slider("Wirksamkeit", 0.0, 10.0, 5.0, 0.5)
                status = st.selectbox("Status", ["planned", "active", "completed"])
                metrics_text = st.text_area("Kennzahlen (Name: Wert)", key="mmetrics")
                if st.form_submit_button("Maßnahme hinzufügen") and title.strip():
                    model.measures.append(
                        Measure(
                            id=new_id("measure"),
                            title=title.strip(),
                            phase_id=phase_id,
                            position=float(position),
                            description=description.strip(),
                            effectiveness=float(effectiveness),
                            status=status,
                            lanes=lanes or ["v1"],
                            metrics=parse_metrics(metrics_text),
                        )
                    )
                    save_model(model)
                    st.rerun()

        st.markdown("---")
        if st.button("Alle Eingaben in JSON speichern", width="stretch"):
            save_model(model)
            st.success("scenario_data.json überschrieben.")

    with right:
        st.subheader("3D-Körper um die zentrale Zeitachse")
        st.plotly_chart(
            build_3d_axis_figure(model),
            width="stretch",
            config={"displayModeBar": True, "scrollZoom": True, "displaylogo": False},
        )

        ev_df = pd.DataFrame(
            [
                {
                    "Szenario": e.title,
                    "Phase": phase_name_map(model).get(e.phase_id, e.phase_id),
                    "Zeit": e.position,
                    "Verläufe": ", ".join(LANES[l]["label"] for l in e.lanes),
                    "Intensität": e.intensity,
                    "Risiko": e.risk,
                }
                for e in sorted(model.events, key=lambda x: x.position)
            ]
        )
        me_df = pd.DataFrame(
            [
                {
                    "Maßnahme": m.title,
                    "Phase": phase_name_map(model).get(m.phase_id, m.phase_id),
                    "Zeit": m.position,
                    "Verläufe": ", ".join(LANES[l]["label"] for l in m.lanes),
                    "Wirksamkeit": m.effectiveness,
                    "Status": m.status,
                }
                for m in sorted(model.measures, key=lambda x: x.position)
            ]
        )
        t1, t2 = st.tabs(["Szenarien", "Gegenmaßnahmen"])
        with t1:
            st.dataframe(ev_df, width="stretch")
        with t2:
            st.dataframe(me_df, width="stretch")


if __name__ == "__main__":
    main()
