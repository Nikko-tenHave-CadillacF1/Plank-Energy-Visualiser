"""Matplotlib + Plotly rendering for the plank energy visualiser."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

from fit import SurfaceFit

CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "blue_cyan_yellow_red",
    [(0.0, "#00008B"), (0.33, "#00CCCC"), (0.67, "#DDDD00"), (1.0, "#8B0000")],
)
PLOTLY_COLORSCALE = [
    [0.0, "rgb(0,0,139)"], [0.17, "rgb(0,100,160)"], [0.33, "rgb(0,204,204)"],
    [0.5, "rgb(128,210,100)"], [0.67, "rgb(221,221,0)"], [0.83, "rgb(180,80,0)"],
    [1.0, "rgb(139,0,0)"],
]


def _grid(h: np.ndarray, k: np.ndarray, fit: SurfaceFit, n: int = 400):
    """Build a fitted surface grid with 5% padding around the data extent."""
    k_pad = max((k.max() - k.min()) * 0.05, 1.0)
    h_pad = max((h.max() - h.min()) * 0.05, 0.1)
    k_grid = np.linspace(k.min() - k_pad, k.max() + k_pad, n)
    h_grid = np.linspace(h.min() - h_pad, h.max() + h_pad, n)
    K, H = np.meshgrid(k_grid, h_grid)
    E = fit.predict(H.ravel(), K.ravel()).reshape(H.shape)
    return k_grid, h_grid, K, H, E


def _color_limits(E_grid: np.ndarray, cutoff: float) -> tuple[float, float]:
    vmin = float(max(0.0, E_grid.min()))
    vmax = float(cutoff)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def render_matplotlib(
    df: pd.DataFrame, fit: SurfaceFit, cutoff: float, out_png: Path,
    *, x_col: str = "KHeaveCPF", y_col: str = "hPlankF", z_col: str = "EPlankF",
    show: bool = True,
) -> None:
    h, k, e = df[y_col].to_numpy(), df[x_col].to_numpy(), df[z_col].to_numpy()
    k_grid, h_grid, K, H, E_plane = _grid(h, k, fit)
    vmin, vmax = _color_limits(E_plane, cutoff)
    levels = np.linspace(vmin, vmax, 13)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(K, H, E_plane, levels=levels, cmap=CUSTOM_CMAP,
                alpha=0.25, vmin=vmin, vmax=vmax, extend="both")
    cs = ax.contour(K, H, E_plane, levels=levels, cmap=CUSTOM_CMAP,
                    alpha=0.9, vmin=vmin, vmax=vmax, linewidths=2)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.1f")
    sc = ax.scatter(k, h, c=e, cmap=CUSTOM_CMAP, edgecolors="black",
                    s=80, zorder=5, vmin=vmin, vmax=vmax)

    fig.colorbar(sc, ax=ax, label=z_col)
    ax.set(xlabel=x_col, ylabel=y_col,
           title=f"{x_col} vs {y_col} \u2013 Fitted {z_col} Contours")

    # Place info box in the corner farthest from data points.
    k_range = k.max() - k.min() or 1.0
    h_range = h.max() - h.min() or 1.0
    k_norm = (k - k.min()) / k_range
    h_norm = (h - h.min()) / h_range
    corners = [(0.02, 0.02, "bottom", "left"),
               (0.98, 0.02, "bottom", "right"),
               (0.02, 0.98, "top", "left"),
               (0.98, 0.98, "top", "right")]
    best = max(corners,
               key=lambda c: np.sqrt((k_norm - c[0])**2 + (h_norm - c[1])**2).min())
    ax.text(best[0], best[1],
            f"Fit ({fit.model}): {fit.equation_text()}\nR\u00b2={fit.r2:.3f}  RMSE={fit.rmse:.2f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment=best[2], horizontalalignment=best[3],
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.9),
                        fontsize=9, visible=False)

    def on_move(event):
        if event.inaxes != ax:
            annot.set_visible(False)
        else:
            kv, hv = event.xdata, event.ydata
            ev = float(fit.predict(np.array([hv]), np.array([kv]))[0])
            annot.xy = (kv, hv)
            annot.set_text(f"{x_col}={kv:.2f}, {y_col}={hv:.2f}\n{z_col}(fit)={ev:.1f}")
            annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def render_plotly(
    df: pd.DataFrame, fit: SurfaceFit, cutoff: float, out_html: Path,
    *, x_col: str = "KHeaveCPF", y_col: str = "hPlankF", z_col: str = "EPlankF",
) -> None:
    h, k, e = df[y_col].to_numpy(), df[x_col].to_numpy(), df[z_col].to_numpy()
    runs = df["run"].to_numpy() if "run" in df.columns else np.arange(len(df)).astype(str)

    k_grid, h_grid, _, _, E_plane = _grid(h, k, fit)
    vmin, vmax = _color_limits(E_plane, cutoff)

    e_fit = fit.predict(h, k)
    resid = e - e_fit
    abs_resid = np.abs(resid)
    sizes = 12.0 + abs_resid * (10.0 / max(abs_resid.max(), 1e-9))

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=k_grid, y=h_grid, z=E_plane, colorscale=PLOTLY_COLORSCALE,
        contours=dict(showlabels=True, labelfont=dict(size=11, color="black")),
        line=dict(width=3), opacity=0.45, zmin=vmin, zmax=vmax,
        colorbar=dict(title=dict(text=f"{z_col} (Fitted)", font=dict(size=13)),
                      tickfont=dict(size=11)),
        hovertemplate=(f"<b>{x_col}</b>: %{{x:.2f}}<br><b>{y_col}</b>: %{{y:.2f}}<br>"
                       f"<b>{z_col} (fitted)</b>: %{{z:.1f}}<extra></extra>"),
        name="Fitted surface",
    ))
    fig.add_trace(go.Scatter(
        x=k, y=h, mode="markers",
        marker=dict(size=sizes, color=e, colorscale=PLOTLY_COLORSCALE,
                    cmin=vmin, cmax=vmax, line=dict(width=1.2, color="black"),
                    showscale=False),
        customdata=np.column_stack([e, e_fit, resid, runs]),
        hovertemplate=(f"<b>Run</b>: %{{customdata[3]}}<br>"
                       f"<b>{x_col}</b>: %{{x:.2f}}<br><b>{y_col}</b>: %{{y:.2f}}<br>"
                       f"<b>{z_col} (actual)</b>: %{{customdata[0]:.1f}}<br>"
                       f"<b>{z_col} (fitted)</b>: %{{customdata[1]:.1f}}<br>"
                       f"<b>Residual</b>: %{{customdata[2]:+.1f}}<extra></extra>"),
        name="Measured data",
    ))

    subtitle = (f"Fit ({fit.model}): {fit.equation_text()}<br>"
                f"R\u00b2={fit.r2:.3f}, RMSE={fit.rmse:.2f}"
                "  \u2014  marker size \u221d |residual|")
    fig.update_layout(
        title=dict(text=f"{x_col} vs {y_col} \u2013 Fitted {z_col} Contours<br><sub>{subtitle}</sub>",
                   x=0.5, font=dict(size=16)),
        xaxis=dict(title=x_col, gridcolor="#ddd", zeroline=False),
        yaxis=dict(title=y_col, gridcolor="#ddd", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1),
        width=950, height=680, margin=dict(l=70, r=40, t=100, b=60),
    )
    fig.write_html(out_html, include_plotlyjs=True)
    print(f"Saved {out_html}")
