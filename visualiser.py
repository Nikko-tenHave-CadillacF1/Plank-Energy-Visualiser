import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

# Custom colormap: dark blue → cyan → yellow → dark red
_cmap_colors = [(0.0, "#00008B"), (0.33, "#00CCCC"), (0.67, "#DDDD00"), (1.0, "#8B0000")]
custom_cmap = LinearSegmentedColormap.from_list(
    "blue_cyan_yellow_red",
    [(s, c) for s, c in _cmap_colors],
)

# Load data
df = pd.read_csv("PlankData.csv")
h = df["hPlankF"].values
k = df["KHeaveCPF"].values
e = df["EPlankF"].values

# Fit a plane: EPlankF = a*hPlankF + b*KHeaveCPF + c
A = np.column_stack([h, k, np.ones_like(h)])
coeffs, _, _, _ = np.linalg.lstsq(A, e, rcond=None)
a, b, c = coeffs

# Create grid for contours using the fitted plane
k_grid = np.linspace(k.min() - 20, k.max() + 20, 500)
h_grid = np.linspace(h.min() - 0.5, h.max() + 0.5, 500)
K, H = np.meshgrid(k_grid, h_grid)
E_plane = a * H + b * K + c

# Plank energy cutoff: values above this are clamped to dark red
E_CUTOFF = 60

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

# Contour lines of equal plank energy from the fitted plane
contour = ax.contour(K, H, E_plane, levels=12, cmap=custom_cmap, alpha=0.9,
                     vmin=E_plane.min(), vmax=E_CUTOFF, linewidths=2)
ax.clabel(contour, inline=True, fontsize=9, fmt="%.1f")

# Filled contours for background colouring
contourf = ax.contourf(K, H, E_plane, levels=12, cmap=custom_cmap, alpha=0.25,
                       vmin=E_plane.min(), vmax=E_CUTOFF)

# Scatter points coloured by actual EPlankF
sc = ax.scatter(k, h, c=e, cmap=custom_cmap, edgecolors="black", s=80, zorder=5,
                vmin=E_plane.min(), vmax=E_CUTOFF)

cbar = fig.colorbar(sc, ax=ax, label="EPlankF (Plank Energy)")

ax.set_xlabel("KHeaveCPF")
ax.set_ylabel("hPlankF")
ax.set_title("KHeaveCPF vs hPlankF – Fitted Plank Energy Contours")

# Annotate fitted plane equation
eq_text = f"Fitted plane: E = {a:.2f}·h + {b:.3f}·K + {c:.1f}"
ax.text(0.02, 0.02, eq_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Interactive cursor: hover to query fitted plank energy
annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.9),
                    fontsize=9, visible=False)


def on_move(event):
    if event.inaxes != ax:
        annot.set_visible(False)
        fig.canvas.draw_idle()
        return
    kv, hv = event.xdata, event.ydata
    ev = a * hv + b * kv + c
    annot.xy = (kv, hv)
    annot.set_text(f"K={kv:.0f}, h={hv:.2f}\nEPlankF(fit)={ev:.1f}")
    annot.set_visible(True)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.tight_layout()
fig.savefig("plank_energy_plot.png", dpi=300)
print("Saved plank_energy_plot.png")
plt.show()

# --- Plotly interactive HTML export ---
blue_cyan_yellow_red = [
    [0.0, "rgb(0,0,139)"], [0.17, "rgb(0,100,160)"], [0.33, "rgb(0,204,204)"],
    [0.5, "rgb(128,210,100)"], [0.67, "rgb(221,221,0)"], [0.83, "rgb(180,80,0)"],
    [1.0, "rgb(139,0,0)"],
]

fig_plotly = go.Figure()

# Filled contours from fitted plane
fig_plotly.add_trace(go.Contour(
    x=k_grid, y=h_grid, z=E_plane,
    colorscale=blue_cyan_yellow_red,
    contours=dict(showlabels=True, labelfont=dict(size=11, color="black")),
    line=dict(width=3),
    opacity=0.45,
    zmin=E_plane.min(), zmax=E_CUTOFF,
    colorbar=dict(title=dict(text="EPlankF (Fitted)", font=dict(size=13)),
                  tickfont=dict(size=11)),
    hovertemplate=(
        "<b>KHeaveCPF</b>: %{x:.0f}<br>"
        "<b>hPlankF</b>: %{y:.2f}<br>"
        "<b>EPlankF (fitted)</b>: %{z:.1f}"
        "<extra></extra>"
    ),
    name="Fitted plane",
))

# Scatter points with hover showing actual and fitted values
e_fitted = a * h + b * k + c
fig_plotly.add_trace(go.Scatter(
    x=k, y=h,
    mode="markers",
    marker=dict(
        size=12,
        color=e,
        colorscale=blue_cyan_yellow_red,
        cmin=E_plane.min(),
        cmax=E_CUTOFF,
        line=dict(width=1.2, color="black"),
        showscale=False,
    ),
    customdata=np.column_stack([e, e_fitted]),
    hovertemplate=(
        "<b>KHeaveCPF</b>: %{x:.0f}<br>"
        "<b>hPlankF</b>: %{y:.2f}<br>"
        "<b>EPlankF (actual)</b>: %{customdata[0]:.1f}<br>"
        "<b>EPlankF (fitted)</b>: %{customdata[1]:.1f}"
        "<extra></extra>"
    ),
    name="Measured data",
))

fig_plotly.update_layout(
    title=dict(
        text=(f"KHeaveCPF vs hPlankF \u2013 Fitted Plank Energy Contours<br>"
              f"<sub>Fitted plane: E = {a:.2f}\u00b7h + {b:.3f}\u00b7K + {c:.1f}</sub>"),
        x=0.5, font=dict(size=16),
    ),
    xaxis=dict(title="KHeaveCPF", gridcolor="#ddd", zeroline=False),
    yaxis=dict(title="hPlankF", gridcolor="#ddd", zeroline=False),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial, sans-serif", size=12, color="#333"),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#ccc", borderwidth=1),
    width=900, height=650,
    margin=dict(l=70, r=40, t=90, b=60),
)

fig_plotly.write_html("plank_energy_plot.html", include_plotlyjs=True)
print("Saved plank_energy_plot.html")
