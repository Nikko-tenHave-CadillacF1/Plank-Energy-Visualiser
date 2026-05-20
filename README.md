# Plank Energy Visualiser

Scatter plot of **hPlankF** vs **KHeaveCPF** with a least-squares surface fit
for **EPlankF** (plank energy). Contours of equal plank energy are overlaid,
coloured dark-blue (low) → cyan → yellow → dark-red (high).

## Workspace layout

```
.
├── data/                       # DLS run parquet files (input)
│   └── <h> EOS hPlank <K> KHeaveCPF_DLS.parquet
├── ingest.py                   # parquet -> PlankData.csv (with mtime cache)
├── fit.py                      # plane / quadratic surface fitting
├── plot.py                     # matplotlib + plotly rendering
├── visualiser.py               # CLI entry point
├── PlankData.csv               # generated per-run summary
├── plank_energy_plot.png       # static output
└── plank_energy_plot.html      # interactive output
```

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python visualiser.py                        # ingest data/, fit plane, plot
python visualiser.py --model quadratic      # quadratic surface in (h, K)
python visualiser.py --rebuild              # ignore the ingest cache
python visualiser.py --cutoff 80 --no-show  # headless run, custom colour cap
```

You can also run the ingester standalone:

```
python ingest.py --data-dir data --csv PlankData.csv
```

## Input requirements

- Parquet filenames must follow `<hPlankF> EOS hPlank <KHeaveCPF> KHeaveCPF_DLS.parquet`,
  e.g. `5p22 EOS hPlank 611 KHeaveCPF_DLS.parquet` → `hPlankF=5.22`, `KHeaveCPF=611`.
- Each parquet must contain `_fzPlankF`, `vCar`, and `UnixTimeMs` columns.

## Computation

```
PPlank_F = 0.001 · max(0.1 · _fzPlankF · vCar/3.6, 0)             [kW]
EPlankF  = ∫ PPlank_F dt at end of run (cumulative_trapezoid)     [kJ]
```

The ingester caches results by hashing parquet filenames, mtimes and sizes
(stored alongside the CSV as `PlankData.csv.sig`); re-runs are instant unless
inputs change.

## Fit reporting

Each run prints the chosen model, its coefficients, R², and RMSE. The same
information is overlaid on both the PNG and HTML plots. The Plotly scatter
sizes points by `|residual|` so outliers stand out at a glance.
