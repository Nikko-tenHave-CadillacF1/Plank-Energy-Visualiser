"""
Plank Energy Visualiser
=======================
Configure the settings below, then run:   python visualiser.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ingest import build_csv, IngestConfig
from fit import fit_surface
from plot import render_matplotlib, render_plotly


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  USER CONFIGURATION – edit the values below to suit your data              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")                # Folder containing .parquet run files
CSV_PATH = Path("PlankData.csv")       # Output / intermediate CSV
REBUILD  = False                       # Set True to force re-ingest (ignore cache)

# ── Axis names (used as CSV headers and plot labels) ──────────────────────────
X_PARAM = "KHeaveCPF"                  # x-axis label
Y_PARAM = "hPlankF"                    # y-axis label
Z_PARAM = "EPlankF"                    # colour-axis label

# ── Axis sources ──────────────────────────────────────────────────────────────
# Each axis value can come from:
#   "filename"  – parsed from the parquet filename (see FILENAME_PATTERN below)
#   "channel"   – average of a parquet data channel at a target speed
#   "energy"    – (z-axis only) the built-in plank energy integral
X_SOURCE = "filename"
Y_SOURCE = "filename"
Z_SOURCE = "energy"

# ── Channel-averaging settings ────────────────────────────────────────────────
# Used when an axis source is "channel". Set SPEED_AT to the target vCar [kph];
# the tool averages the channel values where vCar is within ±1 kph of this.
X_CHANNEL = None                       # e.g. "_fzPlankF"
Y_CHANNEL = None                       # e.g. "someChannelName"
Z_CHANNEL = None                       # e.g. "aeroBalance" (overrides energy calc)
SPEED_AT  = None                       # e.g. 200.0

# ── Filename parsing ──────────────────────────────────────────────────────────
# Regex with named capture groups. Group names must match the axis *_PARAM names
# for any axis whose source is "filename".
# Default pattern matches: "5p22 EOS hPlank 611 KHeaveCPF_DLS.parquet"
FILENAME_PATTERN = None                # None = use built-in default pattern
# Groups where 'p' is used as the decimal separator (e.g. "5p22" → 5.22)
P_DECIMAL_GROUPS = ["hPlankF"]

# ── Fitting ───────────────────────────────────────────────────────────────────
MODEL  = "plane"                       # "plane" or "quadratic"
CUTOFF = 60.0                          # Upper clamp for the colour scale

# ── Output ────────────────────────────────────────────────────────────────────
PNG_PATH  = Path("plank_energy_plot.png")
HTML_PATH = Path("plank_energy_plot.html")
SHOW_PLOT = True                       # Set False for headless / CI runs

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  END OF CONFIGURATION – no need to edit below this line                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def main() -> None:
    config = IngestConfig(
        x_param=X_PARAM,
        y_param=Y_PARAM,
        z_param=Z_PARAM,
        x_source=X_SOURCE,
        y_source=Y_SOURCE,
        z_source=Z_SOURCE,
        x_channel=X_CHANNEL,
        y_channel=Y_CHANNEL,
        z_channel=Z_CHANNEL,
        speed_at=SPEED_AT,
        filename_pattern=FILENAME_PATTERN,
        p_decimal_groups=P_DECIMAL_GROUPS,
    )

    if DATA_DIR.is_dir():
        build_csv(DATA_DIR, CSV_PATH,
                  config=config,
                  use_cache=not REBUILD,
                  verbose=True)
    elif not CSV_PATH.exists():
        raise SystemExit(
            f"No data directory ({DATA_DIR}) and no CSV ({CSV_PATH}). Nothing to plot."
        )

    df = pd.read_csv(CSV_PATH)
    x_col, y_col, z_col = config.x_param, config.y_param, config.z_param

    surface = fit_surface(df[y_col].to_numpy(),
                          df[x_col].to_numpy(),
                          df[z_col].to_numpy(),
                          model=MODEL)
    print(f"Fit model={surface.model}  R²={surface.r2:.3f}  RMSE={surface.rmse:.2f}")
    print(f"   {surface.equation_text()}")

    render_matplotlib(df, surface, CUTOFF, PNG_PATH,
                      x_col=x_col, y_col=y_col, z_col=z_col,
                      show=SHOW_PLOT)
    render_plotly(df, surface, CUTOFF, HTML_PATH,
                  x_col=x_col, y_col=y_col, z_col=z_col)


if __name__ == "__main__":
    main()
