"""Matplotlib + Plotly rendering for the plank energy visualiser."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

from fit import SurfaceFit


_CMAP_STOPS = [(0.0, "#00008B"), (0.33, "#00CCCC"), (0.67, "#DDDD00"), (1.0, "#8B0000")]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("blue_cyan_yellow_red", _CMAP_STOPS)

PLOTLY_COLORSCALE = [
    [0.0, "rgb(0,0,139)"], [0.17, "rgb(0,100,160)"], [0.33, "rgb(0,204,204)"],
    [0.5, "rgb(128,210,100)"], [0.67, "rgb(221,221,0)"], [0.83, "rgb(180,80,0)"],
    [1.0, "rgb(139,0,0)"],
]


def _grid(h: np.ndarray, k: np.ndarray, fit: SurfaceFit, n: int = 400):
    k_grid = np.linspace(k.min() - 20, k.max() + 20, n)
    h_grid = np.linspace(h.min() - 0.5, h.max() + 0.5, n)
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
    df: pd.DataFrame,
    fit: SurfaceFit,
    cutoff: float,
    out_png: Path,
    show: bool = True,
) -> None:
    h = df["hPlankF"].to_numpy()
    k = df["KHeaveCPF"].to_numpy()
    e = df["EPlankF"].to_numpy()

    k_grid, h_grid, K, H, E_plane = _grid(h, k, fit)
    vmin, vmax = _color_limits(E_plane, cutoff)
    levels = np.linspace(vmin, vmax, 13)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(K, H, E_plane, levels=levels, cmap=CUSTOM_CMAP, alpha=0.25,
                vmin=vmin, vmax=vmax, extend="both")
    cs = ax.contour(K, H, E_plane, levels=levels, cmap=CUSTOM_CMAP, alpha=0.9,
                    vmin=vmin, vmax=vmax, linewidths=2)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%.1f")

    sc = ax.scatter(k[1:], h[1:], c=e[1:], cmap=CUSTOM_CMAP, edgecolors="black",
                    s=80, zorder=5, vmin=vmin, vmax=vmax)
    ax.scatter(k[0], h[0], c=[e[0]], cmap=CUSTOM_CMAP, edgecolors="black", s=220,
               zorder=6, vmin=vmin, vmax=vmax, marker="*", linewidths=1.0,
               label="Baseline run")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.colorbar(sc, ax=ax, label="EPlankF (Plank Energy) [kJ]")
    ax.set_xlabel("KHeaveCPF [N/mm]")
    ax.set_ylabel("hPlankF [mm]")
    ax.set_title("KHeaveCPF vs hPlankF \u2013 Fitted Plank Energy Contours")

    info = (f"Fit ({fit.model}): {fit.equation_text()}\n"
            f"R\u00b2 = {fit.r2:.3f},   RMSE = {fit.rmse:.2f} kJ")
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.9),
                        fontsize=9, visible=False)

    def on_move(event):
        if event.inaxes != ax:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        kv, hv = event.xdata, event.ydata
        ev = float(fit.predict(np.array([hv]), np.array([kv]))[0])
        annot.xy = (kv, hv)
        annot.set_text(f"K={kv:.0f}, h={hv:.2f}\nEPlankF(fit)={ev:.1f}")
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
    df: pd.DataFrame,
    fit: SurfaceFit,
    cutoff: float,
    out_html: Path,
) -> None:
    h = df["hPlankF"].to_numpy()
    k = df["KHeaveCPF"].to_numpy()
    e = df["EPlankF"].to_numpy()
    runs = df["run"].to_numpy() if "run" in df.columns else np.array([f"row {i}" for i in range(len(df))])

    k_grid, h_grid, K, H, E_plane = _grid(h, k, fit)
    vmin, vmax = _color_limits(E_plane, cutoff)

    e_fit = fit.predict(h, k)
    resid = e - e_fit
    # Marker size scales with residual magnitude (12 baseline, up to ~22).
    abs_resid = np.abs(resid)
    size_scale = 10.0 / max(abs_resid.max(), 1e-9)
    sizes = 12.0 + abs_resid * size_scale

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=k_grid, y=h_grid, z=E_plane,
        colorscale=PLOTLY_COLORSCALE,
        contours=dict(showlabels=True, labelfont=dict(size=11, color="black")),
        line=dict(width=3),
        opacity=0.45,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title=dict(text="EPlankF (Fitted) [kJ]", font=dict(size=13)),
                      tickfont=dict(size=11)),
        hovertemplate=("<b>KHeaveCPF</b>: %{x:.0f}<br>"
                       "<b>hPlankF</b>: %{y:.2f}<br>"
                       "<b>EPlankF (fitted)</b>: %{z:.1f}<extra></extra>"),
        name="Fitted surface",
    ))

    # Scatter (skip baseline = row 0)
    customdata = np.column_stack([e[1:], e_fit[1:], resid[1:], runs[1:]])
    fig.add_trace(go.Scatter(
        x=k[1:], y=h[1:], mode="markers",
        marker=dict(size=sizes[1:], color=e[1:], colorscale=PLOTLY_COLORSCALE,
                    cmin=vmin, cmax=vmax, line=dict(width=1.2, color="black"),
                    showscale=False),
        customdata=customdata,
        hovertemplate=("<b>Run</b>: %{customdata[3]}<br>"
                       "<b>KHeaveCPF</b>: %{x:.0f}<br>"
                       "<b>hPlankF</b>: %{y:.2f}<br>"
                       "<b>EPlankF (actual)</b>: %{customdata[0]:.1f}<br>"
                       "<b>EPlankF (fitted)</b>: %{customdata[1]:.1f}<br>"
                       "<b>Residual</b>: %{customdata[2]:+.1f}<extra></extra>"),
        name="Measured data",
    ))

    fig.add_trace(go.Scatter(
        x=[k[0]], y=[h[0]], mode="markers",
        marker=dict(size=max(sizes[0], 18), symbol="star",
                    color=[e[0]], colorscale=PLOTLY_COLORSCALE,
                    cmin=vmin, cmax=vmax, line=dict(width=1.5, color="black"),
                    showscale=False),
        customdata=np.array([[e[0], e_fit[0], resid[0], runs[0]]]),
        hovertemplate=("<b>Baseline Run</b>: %{customdata[3]}<br>"
                       "<b>KHeaveCPF</b>: %{x:.0f}<br>"
                       "<b>hPlankF</b>: %{y:.2f}<br>"
                       "<b>EPlankF (actual)</b>: %{customdata[0]:.1f}<br>"
                       "<b>EPlankF (fitted)</b>: %{customdata[1]:.1f}<br>"
                       "<b>Residual</b>: %{customdata[2]:+.1f}<extra></extra>"),
        name="Baseline run",
    ))

    subtitle = (f"Fit ({fit.model}): {fit.equation_text()}<br>"
                f"R\u00b2 = {fit.r2:.3f}, RMSE = {fit.rmse:.2f} kJ"
                "  \u2014  marker size \u221d |residual|")
    fig.update_layout(
        title=dict(
            text=f"KHeaveCPF vs hPlankF \u2013 Fitted Plank Energy Contours<br><sub>{subtitle}</sub>",
            x=0.5, font=dict(size=16),
        ),
        xaxis=dict(title="KHeaveCPF [N/mm]", gridcolor="#ddd", zeroline=False),
        yaxis=dict(title="hPlankF [mm]", gridcolor="#ddd", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1),
        width=950, height=680,
        margin=dict(l=70, r=40, t=100, b=60),
    )

    fig.write_html(out_html, include_plotlyjs=True)
    print(f"Saved {out_html}")
