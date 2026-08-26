"""High-impact Real-time 2D Smiles & Dual 3D Surfaces with Calibration Error Analysis.

Four-panel diagnostic: market reality vs. model prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
from scipy.interpolate import griddata

from options_pricer import OptionsPricer


class RealtimeVolatilityVisualizer:
    """Renders high-grade financial diagnostics: 2D smiles, dual 3D surfaces, calibration error."""

    def __init__(self) -> None:
        plt.ion()
        plt.style.use("dark_background")
        
        # 2x2 grid: top-left = 2D smiles, top-right = SSVI surface, 
        # bottom-left = market surface, bottom-right = error heatmap
        self.fig = plt.figure(figsize=(18, 14), facecolor="#0e1117")
        self.ax_2d = self.fig.add_subplot(221, facecolor="#161b22")
        self.ax_3d_ssvi = self.fig.add_subplot(222, projection="3d", facecolor="#0e1117")
        self.ax_3d_market = self.fig.add_subplot(223, projection="3d", facecolor="#0e1117")
        self.ax_error = self.fig.add_subplot(224, facecolor="#161b22")

        self.elev: float = 28.0
        self.azim: float = -125.0
        self.ax_3d_ssvi.view_init(elev=self.elev, azim=self.azim)
        self.ax_3d_market.view_init(elev=self.elev, azim=self.azim)

        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title("QuantLab Dual-Surface Diagnostic: Market vs. SSVI+rho(theta)")

        plt.show(block=False)

    def update(self, spot: float, slices: List[Dict[str, Any]]) -> None:
        """Draws 2D smiles, SSVI surface, market surface, and calibration error."""
        valid_slices = [s for s in slices if s.get("ssvi_params") is not None and len(s.get("strikes", [])) >= 5]
        if not valid_slices:
            self.fig.canvas.flush_events()
            return

        # Maintain user's 3D rotation
        if self.ax_3d_ssvi.elev is not None and self.ax_3d_ssvi.azim is not None:
            self.elev = self.ax_3d_ssvi.elev
            self.azim = self.ax_3d_ssvi.azim

        self.ax_2d.clear()
        self.ax_3d_ssvi.clear()
        self.ax_3d_market.clear()
        self.ax_error.clear()
        self.ax_3d_ssvi.view_init(elev=self.elev, azim=self.azim)
        self.ax_3d_market.view_init(elev=self.elev, azim=self.azim)

        valid_slices.sort(key=lambda s: s["t"])
        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0.2, 0.95, len(valid_slices)))

        # --- PANEL 1: 2D SMILES ---
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
            self.ax_2d.scatter(strikes, market_ivs, color=c, s=28, alpha=0.9, zorder=5, marker="o")
            self.ax_2d.plot(strikes_dense, iv_dense, color=c, linewidth=2.0, label=f"{exp_str} (T={t_val:.2f}y)", zorder=4)

        self.ax_2d.axvline(spot, color="#00ffcc", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Spot (${spot:.2f})")
        self.ax_2d.set_title("2D Implied Volatility Smiles (Market Quotes vs. SSVI Fit)", fontsize=11, fontweight="bold", color="white")
        self.ax_2d.set_xlabel("Strike ($)", fontsize=9, color="#c9d1d9")
        self.ax_2d.set_ylabel("Implied Volatility (%)", fontsize=9, color="#c9d1d9")
        self.ax_2d.grid(True, linestyle=":", alpha=0.25, color="gray")
        self.ax_2d.tick_params(colors="#c9d1d9")
        self.ax_2d.legend(loc="upper right", fontsize=7.5, framealpha=0.4, facecolor="#161b22")

        # Prepare data for 3D surfaces: mesh grid across all strikes/expiries
        all_strikes = np.concatenate([s["strikes"] for s in valid_slices])
        all_t = np.array([s["t"] for s in valid_slices])
        all_theta = np.array([s["theta"] for s in valid_slices])
        
        # Collect actual market IVs at each (strike, t) point for interpolation
        market_points = []
        for s in valid_slices:
            for strike, iv in zip(s["strikes"], s["market_ivs"]):
                market_points.append((strike, s["t"], iv * 100.0))
        market_points = np.array(market_points) if market_points else np.array([]).reshape(0, 3)

        grid_k = np.linspace(all_strikes.min(), all_strikes.max(), 40)
        grid_t = np.linspace(all_t.min(), all_t.max(), 30)
        K_mesh, T_mesh = np.meshgrid(grid_k, grid_t)
        IV_ssvi_mesh = np.zeros_like(K_mesh)
        IV_market_mesh = np.zeros_like(K_mesh)
        Error_mesh = np.zeros_like(K_mesh)

        # --- COMPUTE BOTH SURFACES ---
        rho_0, rho_infty, lam, eta, gamma = valid_slices[0]["ssvi_params"]

        for i, curr_t in enumerate(grid_t):
            curr_theta = np.interp(curr_t, all_t, all_theta)
            log_m = np.log(grid_k / spot)
            
            # SSVI surface
            w_ssvi = OptionsPricer.ssvi_total_variance(log_m, curr_theta, rho_0, rho_infty, lam, eta, gamma)
            IV_ssvi_mesh[i, :] = np.sqrt(np.maximum(w_ssvi, 0.0) / curr_t) * 100.0
            
            # Market surface (interpolated from observed points)
            if len(market_points) > 0:
                points_2d = market_points[:, :2]
                values = market_points[:, 2]
                IV_market_interp = griddata(
                    points_2d, values,
                    (K_mesh[i, :], T_mesh[i, :]),
                    method="linear",
                    fill_value=np.nan
                )
                IV_market_mesh[i, :] = IV_market_interp
            
            # Error (only where both surfaces have data)
            valid_mask = ~(np.isnan(IV_market_mesh[i, :]) | np.isnan(IV_ssvi_mesh[i, :]))
            Error_mesh[i, valid_mask] = np.abs(IV_market_mesh[i, valid_mask] - IV_ssvi_mesh[i, valid_mask]) / np.maximum(IV_market_mesh[i, valid_mask], 0.1)

        # Clip for display
        IV_ssvi_mesh_clipped = np.clip(IV_ssvi_mesh, 20.0, 75.0)
        IV_market_mesh_clipped = np.clip(IV_market_mesh, 20.0, 75.0)

        # --- PANEL 2: SSVI SURFACE ---
        surf_ssvi = self.ax_3d_ssvi.plot_surface(
            K_mesh, T_mesh, IV_ssvi_mesh_clipped,
            cmap="inferno", edgecolor="#222222", linewidth=0.2, alpha=0.92, antialiased=True,
        )
        self.ax_3d_ssvi.set_title("SSVI+rho(theta) Model Surface", fontsize=11, fontweight="bold", color="white", pad=10)
        self.ax_3d_ssvi.set_xlabel("Strike ($)", fontsize=8, color="#c9d1d9", labelpad=5)
        self.ax_3d_ssvi.set_ylabel("Expiry (T in Years)", fontsize=8, color="#c9d1d9", labelpad=5)
        self.ax_3d_ssvi.set_zlabel("IV (%)", fontsize=8, color="#c9d1d9", labelpad=5)
        self.ax_3d_ssvi.tick_params(colors="#c9d1d9")
        self.ax_3d_ssvi.xaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))
        self.ax_3d_ssvi.yaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))
        self.ax_3d_ssvi.zaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))

        # --- PANEL 3: MARKET SURFACE (INTERPOLATED) ---
        if not np.isnan(IV_market_mesh_clipped).all():
            surf_market = self.ax_3d_market.plot_surface(
                K_mesh, T_mesh, IV_market_mesh_clipped,
                cmap="plasma", edgecolor="#222222", linewidth=0.2, alpha=0.92, antialiased=True,
            )
            self.ax_3d_market.set_title("Market Reality (Interpolated from Observed Quotes)", fontsize=11, fontweight="bold", color="white", pad=10)
        else:
            self.ax_3d_market.text(0.5, 0.5, "Insufficient market data", ha="center", va="center", color="white")
            self.ax_3d_market.set_title("Market Reality (Interpolated from Observed Quotes)", fontsize=11, fontweight="bold", color="white", pad=10)
        
        self.ax_3d_market.set_xlabel("Strike ($)", fontsize=8, color="#c9d1d9", labelpad=5)
        self.ax_3d_market.set_ylabel("Expiry (T in Years)", fontsize=8, color="#c9d1d9", labelpad=5)
        self.ax_3d_market.set_zlabel("IV (%)", fontsize=8, color="#c9d1d9", labelpad=5)
        self.ax_3d_market.tick_params(colors="#c9d1d9")
        self.ax_3d_market.xaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))
        self.ax_3d_market.yaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))
        self.ax_3d_market.zaxis.set_pane_color((0.08, 0.1, 0.14, 1.0))

        # --- PANEL 4: CALIBRATION ERROR HEATMAP ---
        # Show relative error: |IV_market - IV_ssvi| / IV_market as percentage
        error_pct = Error_mesh * 100.0
        error_pct_clipped = np.clip(error_pct, 0, 10.0)  # Cap at 10% for visibility
        
        im = self.ax_error.contourf(K_mesh, T_mesh, error_pct_clipped, levels=20, cmap="RdYlGn_r")
        self.ax_error.set_title("Calibration Error |IV_market - IV_ssvi| / IV_market (%)", fontsize=11, fontweight="bold", color="white")
        self.ax_error.set_xlabel("Strike ($)", fontsize=9, color="#c9d1d9")
        self.ax_error.set_ylabel("Expiry (T in Years)", fontsize=9, color="#c9d1d9")
        self.ax_error.tick_params(colors="#c9d1d9")
        #cbar = plt.colorbar(im, ax=self.ax_error, label="Error (%)")
        #cbar.ax.tick_params(colors="#c9d1d9")
        #cbar.set_label("Error (%)", color="#c9d1d9")

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """Closes visualizer window safely."""
        plt.close(self.fig)