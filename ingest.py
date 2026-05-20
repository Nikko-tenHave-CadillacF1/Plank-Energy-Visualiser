"""Ingest parquet run files and build a summary CSV.

Axis values (x, y, z) can be sourced from:
  - Filename parameters: extracted via a regex with named capture groups.
  - Channel averages at a speed: mean of a parquet column where vCar ≈ target.
  - Energy integral (z only): built-in plank energy calculation.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_FILENAME_RE = re.compile(
    r"^(?P<hPlankF>\d+p\d+)\s+EOS\s+hPlank\s+(?P<KHeaveCPF>\d+)\s+KHeaveCPF_DLS\.parquet$",
    re.IGNORECASE,
)
FORCE_COL = "_fzPlankF"
SPEED_COL = "vCar"
TIME_COL = "UnixTimeMs"
LAP_COL = "_nLap"
LAP_VALUE = 1
SPEED_TOLERANCE = 1.0  # kph


# ─── Configuration ────────────────────────────────────────────────────────────
@dataclass
class IngestConfig:
    """Describes how to extract x, y, z values from each parquet run."""
    x_param: str = "KHeaveCPF"
    y_param: str = "hPlankF"
    z_param: str = "EPlankF"
    x_source: str = "filename"   # "filename" | "channel"
    y_source: str = "filename"   # "filename" | "channel"
    z_source: str = "energy"     # "energy" | "channel"
    x_channel: str | None = None
    y_channel: str | None = None
    z_channel: str | None = None
    speed_at: float | None = None
    filename_pattern: str | None = None
    p_decimal_groups: list[str] = field(default_factory=lambda: ["hPlankF"])

    def compiled_pattern(self) -> re.Pattern:
        if self.filename_pattern:
            return re.compile(self.filename_pattern, re.IGNORECASE)
        return DEFAULT_FILENAME_RE

    def needs_filename(self) -> bool:
        return self.x_source == "filename" or self.y_source == "filename"

    def energy_needed(self) -> bool:
        return self.z_source == "energy"

    def columns_to_load(self) -> list[str] | None:
        """Return the minimal set of parquet columns needed, or None for all."""
        cols: set[str] = set()
        if self.energy_needed():
            cols.update([FORCE_COL, SPEED_COL, TIME_COL, LAP_COL])
        for src, ch in [
            (self.x_source, self.x_channel),
            (self.y_source, self.y_channel),
            (self.z_source, self.z_channel),
        ]:
            if src == "channel" and ch:
                cols.update([ch, SPEED_COL])
        return sorted(cols) if cols else None

    def validate(self) -> None:
        """Raise ValueError if the configuration is inconsistent."""
        for axis, src, ch in [
            ("x", self.x_source, self.x_channel),
            ("y", self.y_source, self.y_channel),
            ("z", self.z_source, self.z_channel),
        ]:
            if src == "channel":
                if not ch:
                    raise ValueError(f"{axis}_source is 'channel' but {axis}_channel is not set.")
                if self.speed_at is None:
                    raise ValueError(f"{axis}_source is 'channel' but speed_at is not set.")


# ─── Core helpers ─────────────────────────────────────────────────────────────

def parse_filename(path: Path, config: IngestConfig) -> dict[str, float] | None:
    """Extract named numeric values from *path* using the configured regex."""
    m = config.compiled_pattern().match(path.name)
    if not m:
        return None
    values: dict[str, float] = {}
    for name, raw in m.groupdict().items():
        if name in config.p_decimal_groups:
            raw = raw.replace("p", ".")
        values[name] = float(raw)
    return values


def channel_average_at_speed(
    df: pd.DataFrame, channel: str, target_speed: float,
) -> float:
    """Mean of *channel* where vCar is within ±SPEED_TOLERANCE of *target_speed*."""
    speed = df[SPEED_COL].to_numpy()
    mask = np.abs(speed - target_speed) <= SPEED_TOLERANCE
    if not mask.any():
        raise ValueError(
            f"No samples within ±{SPEED_TOLERANCE} kph of {target_speed} kph "
            f"(speed range: {speed.min():.1f}–{speed.max():.1f})"
        )
    return float(df[channel].to_numpy()[mask].mean())


def compute_eplank(df: pd.DataFrame) -> float:
    """Compute plank energy [kJ] from the time-series (lap 1 only)."""
    mask = df[LAP_COL].to_numpy() == LAP_VALUE
    if not mask.any():
        raise ValueError(f"No samples with {LAP_COL} == {LAP_VALUE}")
    fz = df[FORCE_COL].to_numpy()[mask]
    v = df[SPEED_COL].to_numpy()[mask]
    t = df[TIME_COL].to_numpy()[mask] / 1000.0
    p_plank = 0.001 * np.maximum(0.1 * fz * (v / 3.6), 0.0)
    return float(cumulative_trapezoid(p_plank, x=t, initial=0.0)[-1])


def _signature(paths: list[Path], config: IngestConfig) -> str:
    """Hash inputs + config so we can skip re-ingest when nothing changed."""
    h = hashlib.sha1(b"ingest-v5\n")
    h.update(json.dumps(config.__dict__, sort_keys=True, default=str).encode())
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p.name}|{st.st_mtime_ns}|{st.st_size}\n".encode())
    return h.hexdigest()


# ─── Main entry point ─────────────────────────────────────────────────────────

def build_csv(
    data_dir: Path,
    csv_path: Path,
    config: IngestConfig | None = None,
    verbose: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Read every parquet in *data_dir*, compute per-run scalars, write CSV."""
    if config is None:
        config = IngestConfig()
    config.validate()

    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    sig_path = csv_path.with_suffix(csv_path.suffix + ".sig")
    sig = _signature(parquets, config)
    if use_cache and csv_path.exists() and sig_path.exists():
        if sig_path.read_text(encoding="utf-8").strip() == sig:
            if verbose:
                print(f"Cache hit: {csv_path} is up-to-date with {data_dir}.")
            return pd.read_csv(csv_path)

    load_cols = config.columns_to_load()
    rows: list[dict[str, Any]] = []

    for path in parquets:
        file_vals: dict[str, float] = {}
        if config.needs_filename():
            parsed = parse_filename(path, config)
            if parsed is None:
                if verbose:
                    print(f"  SKIP (filename not matched): {path.name}")
                continue
            file_vals = parsed

        df = pd.read_parquet(path, columns=load_cols)

        def _get_value(source: str, param: str, channel: str | None) -> float:
            if source == "filename":
                return file_vals[param]
            if source == "energy":
                return compute_eplank(df)
            return channel_average_at_speed(df, channel, config.speed_at)  # type: ignore[arg-type]

        x_val = _get_value(config.x_source, config.x_param, config.x_channel)
        y_val = _get_value(config.y_source, config.y_param, config.y_channel)
        z_val = _get_value(config.z_source, config.z_param, config.z_channel)

        rows.append({config.x_param: x_val, config.y_param: y_val,
                     config.z_param: z_val, "run": path.stem})
        if verbose:
            print(f"  {path.name:55s}  {config.x_param}={x_val:8.2f}  "
                  f"{config.y_param}={y_val:8.2f}  {config.z_param}={z_val:8.2f}")

    if not rows:
        raise RuntimeError(f"No parquet files in {data_dir} matched the expected naming pattern.")

    out = pd.DataFrame(rows, columns=[config.x_param, config.y_param, config.z_param, "run"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    sig_path.write_text(sig, encoding="utf-8")
    if verbose:
        print(f"Wrote {len(out)} rows -> {csv_path}")
    return out


if __name__ == "__main__":
    build_csv(Path("data"), Path("PlankData.csv"), use_cache=False)
