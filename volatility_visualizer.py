"""High-impact Real-time 2D Smiles & 3D Volatility Surface Visualizer.

Engineered with dark theme aesthetics and smooth temporal surface interpolation,
now powered by the arbitrage-free Global SSVI+rho(theta) model with time-dependent skew.
"""

from __future__ import annotations

from typing import Any, Dict, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np

from options_pricer import OptionsPricer


class RealtimeVolatilityVisualizer:
    """Renders high-grade financial 2D/3D volatility diagnostics in real time."""

    def __init__(self) -> None:
        plt.ion()
        plt.style.use("dark_background")
        
        self.fig = plt.figure(figsize=(16, 7.5), facecolor="#0e1117")
        self.ax_2d = self.fig.add_subplot(121, facecolor="#161b22")
        self.ax_3d = self.fig.add_subplot(122, projection="3d", facecolor="#0e1117")

        self.elev: float = 28.0
        self.azim: float = -125.0
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title("QuantLab Real-Time SSVI+rho(theta) Engine")

        plt.show(block=False)

    def update(self, spot: float, slices: List[Dict[str, Any]]) -> None:
        """Draws professional calibrated smiles and continuous 3D surface."""
        valid_slices = [s for s in slices if s.get("ssvi_params") is not None and len(s.get("strikes", [])) >= 5]
        if not valid_slices:
            self.fig.canvas.flush_events()
            return

        # Maintain user's 3D rotation across refreshes
        if self.ax_3d.elev is not None and self.ax_3d.azim is not None:
            self.elev = self.ax_3d.elev
            self.azim = self.ax_3d.azim

        self.ax_2d.clear()
        self.ax_3d.clear()
        self.ax_3d.view_init(elev=self.elev, azim=self.azim)

        # Time-based gradient palette
        valid_slices.sort(key=lambda s: s["t"])
        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0.2, 0.95, len(valid_slices)))

        # --- PANEL 1: 2D SSVI+rho(theta) SMILES ---
        for idx, s in enumerate(valid_slices):
            t_val = s["t"]
            theta = s["theta"]
            exp_str = s["expiry"]
            strikes = np.asarray(s["strikes"], dtype=float)
            market_ivs = np.asarray(s["market_ivs"], dtype=float) * 100.0
            rho_0, rho_infty, lam, eta, gamma = s["ssvi_params"]

            k_dense = np.linspace(np.log(strikes.min() * 0.99 / spot), np.log(strikes.max() * 1.01 / spot), 100)
            strikes_dense = spot * np.exp(k_dense)
            
            w_dense = OptionsPricer.ssvi_total_variance(k_dense, theta, rho_0, rho_infty, lam, eta, gamma)
            iv_dense = np.sqrt(np.maximum(w_dense, 0.0) / t_val) * 100.0

            c = colors[idx]
            self.ax_2d.scatter(strikes, market_ivs, color=c, s=28, alpha=0.9, zorder=5)
            self.ax_2d.plot(strikes_dense, iv_dense, color=c, linewidth=2.0, label=f"{exp_str} (T={t_val:.2f}y)", zorder=4)

        self.ax_2d.axvline(spot, color="#00ffcc", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Spot (${spot:.2f})")
        self.ax_2d.set_title("Implied Volatility Smiles (OTM Quotes vs. SSVI+rho(theta))", fontsize=12, fontweight="bold", color="white")
        self.ax_2d.set_xlabel("Strike ($)", fontsize=10, color="#c9d1d9")
        self.ax_2d.set_ylabel("Implied Volatility (%)", fontsize=10, color="#c9d1d9")
        self.ax_2d.grid(True, linestyle=":", alpha=0.25, color="gray")
        self.ax_2d.tick_params(colors="#c9d1d9")
        self.ax_2d.legend(loc="upper right", fontsize=7.5, framealpha=0.4, facecolor="#161b22")

        # --- PANEL 2: GLOBAL 3D SSVI+rho(theta) SURFACE ---
        # With rho(theta) parametrized exponentially, the skew adapts naturally across
        # the term structure: short-dated (small theta) has strong negative skew (rho ≈ rho_0),
        # long-dated (large theta) has milder skew (rho ≈ rho_infty). The 3D surface
        # emerges from this single, coherent parametrization without any post-hoc interpolation.
        rho_0, rho_infty, lam, eta, gamma = valid_slices[0]["ssvi_params"]
        
        all_strikes = np.concatenate([s["strikes"] for s in valid_slices])
        all_t = np.array([s["t"] for s in valid_slices])
        all_theta = np.array([s["theta"] for s in valid_slices])

        grid_k = np.linspace(all_strikes.min(), all_strikes.max(), 40)
        grid_t = np.linspace(all_t.min(), all_t.max(), 30)
        K_mesh, T_mesh = np.meshgrid(grid_k, grid_t)
        IV_mesh = np.zeros_like(K_mesh)

        for i, curr_t in enumerate(grid_t):
            curr_theta = np.interp(curr_t, all_t, all_theta)
            log_m = np.log(grid_k / spot)
            
            w = OptionsPricer.ssvi_total_variance(log_m, curr_theta, rho_0, rho_infty, lam, eta, gamma)
            IV_mesh[i, :] = np.sqrt(np.maximum(w, 0.0) / curr_t) * 100.0

        IV_mesh = np.clip(IV_mesh, 20.0, 75.0)

        surf = self.ax_3d.plot_surface(
            K_mesh,
            T_mesh,
            IV_mesh,
            cmap="inferno",
            edgecolor="#222222",
            linewidth=0.2,
            alpha=0.92,
            antialiased=True,
        )

        self.ax_3d.set_title("Global Arbitrage-Free SSVI+rho(theta) Surface", fontsize=12, fontweight="bold", color="white", pad=12)
        self.ax_3d.set_xlabel("Strike ($)", fontsize=9, color="#c9d1d9", labelpad=8)
        self.ax_3d.set_ylabel("Expiry (T in Years)", fontsize=9, color="#c9d1d9", labelpad=8)
        self.ax_3d.set_zlabel("SSVI IV (%)", fontsize=9, color="#c9d1d9", labelpad=8)
        self.ax_3d.tick_params(colors="#c9d1d9")

        self.ax_3d.xaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))
        self.ax_3d.yaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))
        self.ax_3d.zaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """Closes visualizer window safely."""
        plt.close(self.fig)