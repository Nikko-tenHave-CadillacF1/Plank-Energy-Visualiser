"""CLI entry point: ingest parquet runs (if present), fit a surface, render plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ingest import build_csv
from fit import fit_surface
from plot import render_matplotlib, render_plotly


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plank energy visualiser.")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Folder of parquet DLS runs. If present, the CSV is (re)built.")
    p.add_argument("--csv", type=Path, default=Path("PlankData.csv"))
    p.add_argument("--cutoff", type=float, default=60.0,
                   help="Upper plank-energy clamp for the colour scale [kJ].")
    p.add_argument("--model", choices=["plane", "quadratic"], default="plane",
                   help="Surface model fitted to (hPlankF, KHeaveCPF) -> EPlankF.")
    p.add_argument("--rebuild", action="store_true",
                   help="Force CSV rebuild from parquet, ignoring the cache.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() (useful for headless runs).")
    p.add_argument("--png", type=Path, default=Path("plank_energy_plot.png"))
    p.add_argument("--html", type=Path, default=Path("plank_energy_plot.html"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_dir.is_dir():
        build_csv(args.data_dir, args.csv,
                  use_cache=not args.rebuild,
                  verbose=True)
    elif not args.csv.exists():
        raise SystemExit(
            f"No data directory ({args.data_dir}) and no CSV ({args.csv}). Nothing to plot."
        )

    df = pd.read_csv(args.csv)
    surface = fit_surface(df["hPlankF"].to_numpy(),
                          df["KHeaveCPF"].to_numpy(),
                          df["EPlankF"].to_numpy(),
                          model=args.model)
    print(f"Fit model={surface.model}  R\u00b2={surface.r2:.3f}  RMSE={surface.rmse:.2f} kJ")
    print(f"   {surface.equation_text()}")

    render_matplotlib(df, surface, args.cutoff, args.png, show=not args.no_show)
    render_plotly(df, surface, args.cutoff, args.html)


if __name__ == "__main__":
    main()
