"""Core 3-D event visualisation for IceCube."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .geometry import load_geometry
from .sql import query_sql


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_to_range(values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Linearly map *values* to [lo, hi]."""
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.full_like(values, (lo + hi) / 2.0, dtype=float)
    return (hi - lo) * (values - vmin) / (vmax - vmin) + lo


def _scale_clipped(
    values: np.ndarray,
    lo: float = 0.0,
    hi: float = 1.0,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> np.ndarray:
    """Scale to [lo, hi] after clipping at percentile bounds."""
    lb, ub = np.percentile(values, [lower_pct, upper_pct])
    clipped = np.clip(values, lb, ub)
    return _scale_to_range(clipped, lo, hi)


def _marker_size_from_charge(
    charge: np.ndarray,
    lower_clip: float = 4.0,
    upper_clip: float = 22.0,
) -> np.ndarray:
    """Log-scale marker sizes clipped to a reasonable range."""
    sizes = np.log2(charge + 1) * 3.0
    return np.clip(sizes, lower_clip, upper_clip)


def _direction_arrow(truth: pd.DataFrame, event: pd.DataFrame):
    """Return two Scatter3d traces: the direction line and a cone arrowhead."""
    zen = float(truth["zenith"].iloc[0])
    azi = float(truth["azimuth"].iloc[0])

    dx = np.sin(zen) * np.cos(azi)
    dy = np.sin(zen) * np.sin(azi)
    dz = np.cos(zen)

    cx = float(event["dom_x"].mean())
    cy = float(event["dom_y"].mean())
    cz = float(event["dom_z"].mean())

    length = 600.0
    x0, y0, z0 = cx - dx * length, cy - dy * length, cz - dz * length
    x1, y1, z1 = cx + dx * length, cy + dy * length, cz + dz * length

    line_trace = go.Scatter3d(
        x=[x0, x1], y=[y0, y1], z=[z0, z1],
        mode="lines",
        line=dict(color="orange", width=4),
        name="Direction",
        hoverinfo="skip",
    )

    # Small cone at the tip to indicate direction
    cone_trace = go.Cone(
        x=[x1], y=[y1], z=[z1],
        u=[dx], v=[dy], w=[dz],
        sizemode="absolute",
        sizeref=80,
        colorscale=[[0, "orange"], [1, "orange"]],
        showscale=False,
        name="",
        hoverinfo="skip",
    )

    return line_trace, cone_trace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_event_scatter(
    event: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    size_scale: float = 1.0,
    use_earliest_time: bool = True,
    colorscale: str = "Turbo",
    show_direction_arrow: bool = True,
    show_first_hit: bool = True,
) -> tuple[go.Scatter3d, list]:
    """Build the Plotly traces for a single event.

    Parameters
    ----------
    event:
        Pulse DataFrame with columns dom_x/y/z, dom_time, charge.
    truth:
        Truth DataFrame (must contain zenith, azimuth, position_x/y/z).
    size_scale:
        Multiply all marker sizes by this factor.
    use_earliest_time:
        Colour by earliest hit time per DOM (True) or charge-weighted mean (False).
    colorscale:
        Plotly colorscale name for the time encoding.
    show_direction_arrow:
        Overlay direction line + cone.
    show_first_hit:
        Mark the earliest-hit DOM with a diamond.

    Returns
    -------
    event_trace : go.Scatter3d
    extras : list[go.BaseTraceType]
        Additional traces (arrow, first-hit marker).
    """
    # Aggregate pulses per DOM
    grouped = event.groupby(["dom_x", "dom_y", "dom_z"], sort=False).agg(
        sum_charge=("charge", "sum"),
        earliest_time=("dom_time", "min"),
        weighted_mean_time=("dom_time", lambda x: (
            (x * event.loc[x.index, "charge"]).sum()
            / event.loc[x.index, "charge"].sum()
        )),
    ).reset_index()

    time_col = "earliest_time" if use_earliest_time else "weighted_mean_time"
    t_scaled = _scale_clipped(grouped[time_col].values)

    sizes = _marker_size_from_charge(grouped["sum_charge"].values) * size_scale

    hover_text = [
        f"x={row.dom_x:.1f}, y={row.dom_y:.1f}, z={row.dom_z:.1f}<br>"
        f"charge={row.sum_charge:.2f}<br>"
        f"time={row[time_col]:.1f} ns"
        for _, row in grouped.iterrows()
    ]

    event_trace = go.Scatter3d(
        x=grouped["dom_x"],
        y=grouped["dom_y"],
        z=grouped["dom_z"],
        mode="markers",
        marker=dict(
            color=t_scaled,
            size=sizes,
            colorscale=colorscale,
            reversescale=True,
            opacity=1.0,
            line=dict(width=0),
            colorbar=dict(
                title="Time",
                thickness=16,
                len=0.6,
                tickvals=[0, 0.5, 1],
                ticktext=["early", "", "late"],
            ),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name="Pulses",
    )

    extras = []

    if show_direction_arrow:
        try:
            line_trace, cone_trace = _direction_arrow(truth, event)
            extras.extend([line_trace, cone_trace])
        except (KeyError, TypeError):
            pass  # truth doesn't have direction info

    if show_first_hit:
        first = event.loc[event["dom_time"].idxmin()]
        extras.append(go.Scatter3d(
            x=[first["dom_x"]],
            y=[first["dom_y"]],
            z=[first["dom_z"]],
            mode="markers+text",
            text=["First hit"],
            textposition="top center",
            marker=dict(size=12, color="cyan", symbol="diamond", line=dict(width=1, color="white")),
            name="First hit",
            hoverinfo="skip",
        ))

    return event_trace, extras


def _build_title(truth: pd.DataFrame, event: pd.DataFrame) -> str:
    """Compose a multi-line title from truth columns."""
    parts = []
    for col in truth.columns:
        val = truth[col].iloc[0]
        try:
            parts.append(f"{col}: {float(val):.3g}")
        except (TypeError, ValueError):
            parts.append(f"{col}: {val}")
    parts.append(f"N pulses: {len(event):.0f}")
    return "<br>".join(parts)


def show_event(
    event: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    mode: str = "show",
    save_path: str | None = None,
    save_format: str = "html",
    width: int = 900,
    height: int = 800,
    geometry_path=None,
    detector_marker_size: int = 3,
    detector_opacity: float = 0.12,
    size_scale: float = 1.0,
    colorscale: str = "Turbo",
    show_direction_arrow: bool = True,
    show_first_hit: bool = True,
    show_title: bool = True,
    show_colorbar: bool = True,
    camera_eye: dict | None = None,
) -> go.Figure:
    """Visualise a single IceCube event.

    Parameters
    ----------
    event:
        Pulse DataFrame.
    truth:
        Truth DataFrame.
    mode:
        ``"show"``  – open in browser / notebook
        ``"save"``  – write to file (requires *save_path*)
        ``"return"`` – return figure without displaying
    save_path:
        File path (without extension) when *mode* is ``"save"``.
    save_format:
        ``"html"`` (default), ``"pdf"``, ``"png"``, ``"svg"``…
    width / height:
        Figure dimensions in pixels.
    geometry_path:
        Custom sensor geometry CSV.  ``None`` uses the bundled file.
    detector_marker_size:
        Size of inactive DOM markers.
    detector_opacity:
        Opacity of the detector array.
    size_scale:
        Scale factor for pulse markers.
    colorscale:
        Plotly colorscale for time encoding.
    show_direction_arrow:
        Overlay a direction line + cone.
    show_first_hit:
        Mark the earliest hit DOM.
    show_title:
        Display truth info as title.
    show_colorbar:
        Show the time colorbar.
    camera_eye:
        Dict with x/y/z for the initial camera position.

    Returns
    -------
    go.Figure
    """
    if mode == "save" and save_path is None:
        raise ValueError("save_path must be provided when mode='save'")

    _, detector_trace = load_geometry(
        path=geometry_path,
        marker_size=detector_marker_size,
        opacity=detector_opacity,
    )

    event_trace, extras = make_event_scatter(
        event,
        truth,
        size_scale=size_scale,
        colorscale=colorscale,
        show_direction_arrow=show_direction_arrow,
        show_first_hit=show_first_hit,
    )

    fig = go.Figure(data=[detector_trace, event_trace] + extras)

    title_text = _build_title(truth, event) if show_title else ""

    default_eye = dict(x=1.6, y=1.6, z=0.8)
    eye = camera_eye or default_eye

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=11), x=0.01, xanchor="left"),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=eye),
            bgcolor="rgba(10,10,20,1)",
        ),
        paper_bgcolor="rgba(15,15,25,1)",
        plot_bgcolor="rgba(15,15,25,1)",
        font=dict(color="white"),
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50),
    )

    if not show_colorbar:
        fig.update_traces(marker_showscale=False)

    if mode == "show":
        fig.show()
    elif mode == "save":
        if save_format == "html":
            fig.write_html(f"{save_path}.html")
        else:
            fig.write_image(f"{save_path}.{save_format}", format=save_format)

    return fig


def show_sql_event(
    db_path: str,
    event_no: int,
    *,
    pulsemap: str = "SRTInIcePulses",
    truth_table: str = "truth",
    truth_cols: list[str] | None = None,
    **show_kwargs,
) -> tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    """Load an event from SQLite and visualise it.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    event_no:
        Event number to query.
    pulsemap:
        Pulse table name.
    truth_table:
        Truth table name.
    truth_cols:
        Subset of truth columns to display in the title.
    **show_kwargs:
        Forwarded to :func:`show_event`.

    Returns
    -------
    fig, truth, event
    """
    event, truth = query_sql(
        db_path,
        event_no,
        pulsemap=pulsemap,
        truth_table=truth_table,
        truth_cols=truth_cols,
    )

    fig = show_event(event, truth, **show_kwargs)
    return fig, truth, event
