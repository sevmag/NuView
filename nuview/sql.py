"""SQLite querying utilities for IceCube event databases."""

import sqlite3
import pandas as pd


def query_sql(
    db_path: str,
    event_no: int,
    pulsemap: str = "SRTInIcePulses",
    truth_table: str = "truth",
    truth_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query pulses and truth for a single event from an SQLite database.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    event_no:
        The ``event_no`` to retrieve.
    pulsemap:
        Name of the pulse table.
    truth_table:
        Name of the truth table.
    truth_cols:
        Which truth columns to return.  ``None`` returns all columns.
        The columns required for plotting (position, zenith/azimuth, IDs)
        are always included.

    Returns
    -------
    event : pd.DataFrame
        Pulse-level data (dom_x, dom_y, dom_z, dom_time, charge, …).
    truth : pd.DataFrame
        Event-level truth information.
    """
    _ALWAYS_INCLUDE = {
        "position_x", "position_y", "position_z",
        "azimuth", "zenith",
        "RunID", "EventID", "SubEventID",
    }

    with sqlite3.connect(db_path) as conn:
        event = pd.read_sql(
            f"SELECT * FROM {pulsemap} WHERE event_no = {event_no}",
            conn,
        ).reset_index(drop=True)

        truth = pd.read_sql(
            f"SELECT * FROM {truth_table} WHERE event_no = {event_no}",
            conn,
        ).reset_index(drop=True)

    if len(event) == 0:
        raise ValueError(f"No pulses found for event_no={event_no} in '{pulsemap}'")
    if len(truth) == 0:
        raise ValueError(f"No truth found for event_no={event_no} in '{truth_table}'")

    if truth_cols is not None:
        keep = list(_ALWAYS_INCLUDE | set(truth_cols))
        keep = [c for c in keep if c in truth.columns]
        truth = truth[keep]

    return event, truth
