# Plank Energy Visualiser

Scatter plot of configurable x/y parameters with a least-squares surface fit
for a z-axis value (colour). Contours of equal z are overlaid, coloured
dark-blue (low) → cyan → yellow → dark-red (high).

By default: **KHeaveCPF** (x) vs **hPlankF** (y), coloured by **EPlankF**
(plank energy computed from the time-series data).

## Workspace layout

```
.
├── data/                       # DLS run parquet files (input)
│   └── <h> EOS hPlank <K> KHeaveCPF_DLS.parquet
├── ingest.py                   # parquet -> PlankData.csv (with mtime cache)
├── fit.py                      # plane / quadratic surface fitting
├── plot.py                     # matplotlib + plotly rendering
├── visualiser.py               # ← CONFIGURE AND RUN THIS FILE
├── PlankData.csv               # generated per-run summary
├── plank_energy_plot.png       # static output
└── plank_energy_plot.html      # interactive output
```

## Setup

```
pip install -r requirements.txt
```

## Usage

1. Open `visualiser.py` and edit the **USER CONFIGURATION** section at the top.
2. Run:
   ```
   python visualiser.py
   ```

That's it. All settings (data paths, axis sources, channel names, fit model,
output files) are configured as plain variables inside `visualiser.py`.

## Configuration reference

All settings live at the top of `visualiser.py`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_DIR` | `data` | Folder containing `.parquet` run files |
| `CSV_PATH` | `PlankData.csv` | Output CSV path |
| `REBUILD` | `False` | Force re-ingest (ignore mtime cache) |
| `X_PARAM` | `"KHeaveCPF"` | x-axis label / CSV column name |
| `Y_PARAM` | `"hPlankF"` | y-axis label / CSV column name |
| `Z_PARAM` | `"EPlankF"` | colour-axis label / CSV column name |
| `X_SOURCE` | `"filename"` | `"filename"` or `"channel"` |
| `Y_SOURCE` | `"filename"` | `"filename"` or `"channel"` |
| `Z_SOURCE` | `"energy"` | `"energy"` or `"channel"` |
| `X_CHANNEL` | `None` | Parquet column name for x (when source = channel) |
| `Y_CHANNEL` | `None` | Parquet column name for y (when source = channel) |
| `Z_CHANNEL` | `None` | Parquet column name for z (when source = channel) |
| `SPEED_AT` | `None` | Target vCar [kph] for channel averaging (±1 kph) |
| `FILENAME_PATTERN` | `None` | Custom regex with named groups (None = built-in) |
| `P_DECIMAL_GROUPS` | `["hPlankF"]` | Groups using `p` as decimal separator |
| `MODEL` | `"plane"` | `"plane"` or `"quadratic"` surface fit |
| `CUTOFF` | `60.0` | Upper clamp for colour scale |
| `SHOW_PLOT` | `True` | Show interactive matplotlib window |

### Axis source modes

- **`"filename"`** – value is parsed from the parquet filename via a regex
  with named capture groups. The group name must match the axis `*_PARAM`.
- **`"channel"`** – value is the mean of the named parquet column where
  `vCar` is within ±1 kph of `SPEED_AT`.
- **`"energy"`** (z-axis only) – uses the built-in plank energy integral.

## Input requirements

- By default, parquet filenames must follow
  `<hPlankF> EOS hPlank <KHeaveCPF> KHeaveCPF_DLS.parquet`,
  e.g. `5p22 EOS hPlank 611 KHeaveCPF_DLS.parquet` → `hPlankF=5.22`, `KHeaveCPF=611`.
  Set `FILENAME_PATTERN` to override this.
- For the default energy calculation: each parquet must contain `_fzPlankF`,
  `vCar`, `UnixTimeMs`, and `_nLap` columns.
- For channel averaging: each parquet must contain `vCar` and the requested
  channel column(s).

## Computation (default z-axis)

```
PPlank_F = 0.001 · max(0.1 · _fzPlankF · vCar/3.6, 0)             [kW]
EPlankF  = ∫ PPlank_F dt at end of run (cumulative_trapezoid)     [kJ]
```

The ingester caches results by hashing parquet filenames, mtimes, sizes, and
the full configuration (stored as `PlankData.csv.sig`); re-runs are instant
unless inputs or settings change.

## Fit reporting

Each run prints the chosen model, its coefficients, R², and RMSE. The same
information is overlaid on both the PNG and HTML plots. The Plotly scatter
sizes points by `|residual|` so outliers stand out at a glance.
