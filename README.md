# Plank Energy Visualiser

Scatter plot of **hPlankF** vs **KHeaveCPF** with a least-squares fitted plane for **EPlankF**. Contours of equal plank energy are overlaid, coloured green (low) to red (high).

## Outputs

- `plank_energy_plot.png` — static image (matplotlib)
- `plank_energy_plot.html` — interactive plot (Plotly), open in any browser to hover and query fitted plank energy at any point

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python visualiser.py
```

Place `PlankData.csv` in the same directory. The CSV must contain columns `hPlankF`, `KHeaveCPF`, and `EPlankF`.
