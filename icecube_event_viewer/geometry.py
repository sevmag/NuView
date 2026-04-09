"""IceCube detector geometry utilities."""

from importlib.resources import files
import pandas as pd
import plotly.graph_objects as go


_DEFAULT_GEOMETRY_PATH = files("icecube_event_viewer.data").joinpath("sensor_geometry.csv")


def load_geometry(
    path=None,
    marker_size: int = 3,
    color: str = "grey",
    opacity: float = 0.15,
) -> tuple[pd.DataFrame, go.Scatter3d]:
    """Load the IceCube detector geometry.

    Parameters
    ----------
    path:
        Path to a custom sensor_geometry.csv. Defaults to the bundled one.
    marker_size:
        Marker size for inactive DOMs.
    color:
        Marker color for inactive DOMs.
    opacity:
        Opacity of the detector array markers.

    Returns
    -------
    geometry_table : pd.DataFrame
        Table with columns ``sensor_id``, ``x``, ``y``, ``z``.
    detector_trace : go.Scatter3d
        Plotly trace of the full detector.
    """
    csv_path = str(path or _DEFAULT_GEOMETRY_PATH)
    geometry_table = pd.read_csv(csv_path)

    detector_trace = go.Scatter3d(
        x=geometry_table["x"],
        y=geometry_table["y"],
        z=geometry_table["z"],
        mode="markers",
        marker=dict(size=marker_size, color=color, opacity=opacity),
        name="Detector",
        hoverinfo="skip",
    )

    return geometry_table, detector_trace
