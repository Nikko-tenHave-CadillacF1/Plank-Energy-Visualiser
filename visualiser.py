import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go

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
h_grid = np.linspace(h.min() - 0.5, h.max() + 0.5, 200)
k_grid = np.linspace(k.min() - 20, k.max() + 20, 200)
H, K = np.meshgrid(h_grid, k_grid)
E_plane = a * H + b * K + c

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

# Contour lines of equal plank energy from the fitted plane
contour = ax.contour(H, K, E_plane, levels=12, cmap="RdYlGn_r", alpha=0.7)
ax.clabel(contour, inline=True, fontsize=8, fmt="%.1f")

# Filled contours for background colouring
contourf = ax.contourf(H, K, E_plane, levels=12, cmap="RdYlGn_r", alpha=0.25)

# Scatter points coloured by actual EPlankF
sc = ax.scatter(h, k, c=e, cmap="RdYlGn_r", edgecolors="black", s=80, zorder=5,
                vmin=E_plane.min(), vmax=E_plane.max())

cbar = fig.colorbar(sc, ax=ax, label="EPlankF (Plank Energy)")

ax.set_xlabel("hPlankF")
ax.set_ylabel("KHeaveCPF")
ax.set_title("hPlankF vs KHeaveCPF – Fitted Plank Energy Contours")

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
    hv, kv = event.xdata, event.ydata
    ev = a * hv + b * kv + c
    annot.xy = (hv, kv)
    annot.set_text(f"h={hv:.2f}, K={kv:.0f}\nEPlankF(fit)={ev:.1f}")
    annot.set_visible(True)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.tight_layout()
fig.savefig("plank_energy_plot.png", dpi=150)
print("Saved plank_energy_plot.png")
plt.show()

# --- Plotly interactive HTML export ---
rdylgn_r = [[0, "rgb(0,104,55)"], [0.1, "rgb(26,152,80)"], [0.2, "rgb(102,189,99)"],
            [0.3, "rgb(166,217,106)"], [0.4, "rgb(217,239,139)"], [0.5, "rgb(255,255,191)"],
            [0.6, "rgb(254,224,139)"], [0.7, "rgb(253,174,97)"], [0.8, "rgb(244,109,67)"],
            [0.9, "rgb(215,48,39)"], [1.0, "rgb(165,0,38)"]]

fig_plotly = go.Figure()

# Filled contours from fitted plane
fig_plotly.add_trace(go.Contour(
    x=h_grid, y=k_grid, z=E_plane,
    colorscale=rdylgn_r,
    contours=dict(showlabels=True, labelfont=dict(size=10, color="black")),
    opacity=0.35,
    colorbar=dict(title=dict(text="EPlankF (Fitted)", font=dict(size=13)),
                  tickfont=dict(size=11)),
    hovertemplate=(
        "<b>hPlankF</b>: %{x:.2f}<br>"
        "<b>KHeaveCPF</b>: %{y:.0f}<br>"
        "<b>EPlankF (fitted)</b>: %{z:.1f}"
        "<extra></extra>"
    ),
    name="Fitted plane",
))

# Scatter points with hover showing actual and fitted values
e_fitted = a * h + b * k + c
fig_plotly.add_trace(go.Scatter(
    x=h, y=k,
    mode="markers",
    marker=dict(
        size=12,
        color=e,
        colorscale=rdylgn_r,
        cmin=E_plane.min(),
        cmax=E_plane.max(),
        line=dict(width=1.2, color="black"),
        showscale=False,
    ),
    customdata=np.column_stack([e, e_fitted]),
    hovertemplate=(
        "<b>hPlankF</b>: %{x:.2f}<br>"
        "<b>KHeaveCPF</b>: %{y:.0f}<br>"
        "<b>EPlankF (actual)</b>: %{customdata[0]:.1f}<br>"
        "<b>EPlankF (fitted)</b>: %{customdata[1]:.1f}"
        "<extra></extra>"
    ),
    name="Measured data",
))

fig_plotly.update_layout(
    title=dict(
        text=(f"hPlankF vs KHeaveCPF \u2013 Fitted Plank Energy Contours<br>"
              f"<sub>Fitted plane: E = {a:.2f}\u00b7h + {b:.3f}\u00b7K + {c:.1f}</sub>"),
        x=0.5, font=dict(size=16),
    ),
    xaxis=dict(title="hPlankF", gridcolor="#ddd", zeroline=False),
    yaxis=dict(title="KHeaveCPF", gridcolor="#ddd", zeroline=False),
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
