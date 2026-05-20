"""Ingest DLS run parquet files and build PlankData.csv.

Each parquet filename is expected to follow the pattern::

    <hPlankF> EOS hPlank <KHeaveCPF> KHeaveCPF_DLS.parquet

where ``hPlankF`` uses ``p`` as a decimal separator (e.g. ``5p22`` -> 5.22) and
``KHeaveCPF`` is an integer. Both are setup parameters and are taken straight
from the filename.

The parquet must contain time-series columns ``_fzPlankF`` (plank vertical
force, N), ``vCar`` (car speed, km/h), and ``UnixTimeMs`` (millisecond
timestamps, used to compute dt).

For each run we compute::

    PPlank_F = 0.001 * max(0.1 * _fzPlankF * (vCar / 3.6), 0)      [kW]
    EPlankF  = trapz(PPlank_F, dt) at end of run                   [kJ]

and write one CSV row per run.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


FILENAME_RE = re.compile(
    r"^(?P<h>\d+p\d+)\s+EOS\s+hPlank\s+(?P<k>\d+)\s+KHeaveCPF_DLS\.parquet$",
    re.IGNORECASE,
)
FORCE_COL = "_fzPlankF"
SPEED_COL = "vCar"
TIME_COL = "UnixTimeMs"  # milliseconds
LAP_COL = "_nLap"
LAP_VALUE = 1


@dataclass
class RunResult:
    run: str
    hPlankF: float
    KHeaveCPF: float
    EPlankF: float


def parse_filename(path: Path) -> tuple[float, float] | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    h = float(m.group("h").replace("p", "."))
    k = float(m.group("k"))
    return h, k


def compute_eplank(df: pd.DataFrame) -> float:
    mask = df[LAP_COL].to_numpy() == LAP_VALUE
    if not mask.any():
        raise ValueError(f"No samples found with {LAP_COL} == {LAP_VALUE}")
    fz = df[FORCE_COL].to_numpy()[mask]
    v = df[SPEED_COL].to_numpy()[mask]
    t = df[TIME_COL].to_numpy()[mask] / 1000.0  # ms -> s
    p_plank = 0.001 * np.maximum(0.1 * fz * (v / 3.6), 0.0)
    e_plank = cumulative_trapezoid(p_plank, x=t, initial=0.0)
    return float(e_plank[-1])


def _signature(paths: list[Path]) -> str:
    """Hash filenames + mtimes + sizes so we can skip re-ingest when nothing changed."""
    h = hashlib.sha1()
    # Bump this tag whenever the ingest semantics change (formula, filters, ...).
    h.update(b"ingest-v3-lap1\n")
    for p in sorted(paths):
        st = p.stat()
        h.update(p.name.encode("utf-8"))
        h.update(f"|{int(st.st_mtime_ns)}|{st.st_size}\n".encode("utf-8"))
    return h.hexdigest()


def build_csv(
    data_dir: Path,
    csv_path: Path,
    verbose: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Read every parquet in *data_dir*, compute per-run scalars, write to *csv_path*."""
    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    sig_path = csv_path.with_suffix(csv_path.suffix + ".sig")
    sig = _signature(parquets)
    if use_cache and csv_path.exists() and sig_path.exists():
        if sig_path.read_text(encoding="utf-8").strip() == sig:
            if verbose:
                print(f"Cache hit: {csv_path} is up-to-date with {data_dir}.")
            return pd.read_csv(csv_path)

    rows: list[RunResult] = []
    for path in parquets:
        parsed = parse_filename(path)
        if parsed is None:
            if verbose:
                print(f"  SKIP (filename pattern not matched): {path.name}")
            continue
        h, k = parsed
        df = pd.read_parquet(path, columns=[FORCE_COL, SPEED_COL, TIME_COL, LAP_COL])
        e = compute_eplank(df)
        rows.append(RunResult(run=path.stem, hPlankF=h, KHeaveCPF=k, EPlankF=e))
        if verbose:
            print(f"  {path.name:55s}  h={h:5.2f}  K={k:6.0f}  E={e:6.1f}")

    if not rows:
        raise RuntimeError(f"No parquet files in {data_dir} matched the expected naming pattern.")

    out = pd.DataFrame([r.__dict__ for r in rows],
                       columns=["hPlankF", "KHeaveCPF", "EPlankF", "run"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    sig_path.write_text(sig, encoding="utf-8")
    if verbose:
        print(f"Wrote {len(out)} rows -> {csv_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--csv", type=Path, default=Path("PlankData.csv"))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    build_csv(args.data_dir, args.csv, use_cache=not args.no_cache)
