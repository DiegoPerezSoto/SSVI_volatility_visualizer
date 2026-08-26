"""Console rendering layer for the trading terminal.

Pure presentation logic: knows how to lay out numbers but nothing about
pricing, IBKR, or trading logic.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import pandas as pd


class TerminalUI:
    """Renders the trading terminal to stdout."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def clear_screen(self) -> None:
        """Clears the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def render(
        self,
        current_price: float,
        open_price: float,
        open_date: str,
        expiries: List[str],
        radar_data: Dict[str, pd.DataFrame],
        net_liquidation: float,
        available_funds: float,
        excess_liquidity: float,
        total_subscriptions: int,
    ) -> None:
        """Draws the full terminal snapshot: header and options radar.

        Args:
            current_price: Current underlying price.
            open_price: Opening price of the day.
            open_date: Date of the open price.
            expiries: List of tracked expiry strings.
            radar_data: Mapping of expiry -> DataFrame with radar metrics.
            net_liquidation: Total account equity.
            available_funds: Available buying power.
            excess_liquidity: Excess liquidity.
            total_subscriptions: Number of active market-data subscriptions.
        """
        self.clear_screen()
        change = current_price - open_price
        change_pct = (change / open_price * 100) if open_price > 0 else 0

        print("=" * 130)
        print(f"    Volatility Visualizer | IBKR TERMINAL  |  CLOCK: {datetime.now().strftime('%H:%M:%S.%f')[:-4]}")
        print("=" * 130)
        print(f" [ ASSET TRACKER: {self.symbol} ]")
        print(f" CURRENT: ${current_price:,.2f} ({change_pct:+.2f}%)   |   OPEN: ${open_price:,.2f} ({open_date})")
        print("-" * 130)
        print(" [ ACCOUNT INFO ]")
        print(
            f" TOTAL EQUITY: ${net_liquidation:,.2f} | "
            f"AVAILABLE: ${available_funds:,.2f} | "
            f"EXCESS LIQ: ${excess_liquidity:,.2f}"
        )
        print(f" ACTIVE SUBSCRIPTIONS: {total_subscriptions}")
        print("-" * 130)

        self._render_radar(radar_data)

        print("=" * 130)
        print(" [ SYSTEM LOG ] -> STREAMING MODE ACTIVE")

    def _render_radar(self, radar_data: Dict[str, pd.DataFrame]) -> None:
        """Renders the options radar tables."""
        if not radar_data:
            print(" Waiting for market open or initial data...")
            return

        for expiry, df in radar_data.items():
            print(f"\n [ OPTION RADAR | EXPIRY: {expiry} ]")
            print("═" * 130)
            print(df.to_string(index=False))
            print("═" * 130)
