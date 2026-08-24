"""Top-level orchestrator: owns IBKR connection and drives the main event loop.

Wires together SubscriptionManager, PortfolioManager, VolatilityManager, TerminalUI, and RealtimeVolatilityVisualizer.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from ib_insync import IB, Stock, Ticker

from portfolio_manager import PortfolioManager
from subscription_manager import SubscriptionManager
from terminal_ui import TerminalUI
from volatility_manager import VolatilityManager
from volatility_visualizer import RealtimeVolatilityVisualizer

logger = logging.getLogger(__name__)

_UNDERLYING_OWNER = "underlying"
_LOOP_INTERVAL_SEC = 0.5
_STARTUP_SETTLE_SEC = 2


class TradingBot:
    """Connects to IBKR, wires up the engine components, and runs the terminal loop."""

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        symbol: str,
        risk_free_rate: float,
        div_yield: float,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.div_yield = div_yield

        self.open_price: float = 0.0
        self.open_date: str = "N/A"
        self.underlying: Optional[Ticker] = None

        self.ib = IB()
        self.ui = TerminalUI(symbol)
        self.visualizer = RealtimeVolatilityVisualizer()

        self.subscriptions: Optional[SubscriptionManager] = None
        self.portfolio: Optional[PortfolioManager] = None
        self.volatility: Optional[VolatilityManager] = None

    def connect(self) -> None:
        """Opens the IBKR connection, attaches event listeners, and constructs engine components."""
        logger.info("Connecting to IBKR at %s:%s (client_id=%s)...", self.host, self.port, self.client_id)
        try:
            self.ib.connect(self.host, self.port, self.client_id, timeout=15)
            
            # Intercept API limit rejections (Error 100: Max number of tickers reached)
            def on_error(req_id: int, error_code: int, error_string: str, contract: object) -> None:
                if error_code == 100:
                    logger.warning(
                        "DATA LIMIT REJECT: IBKR blocked subscription (Code 100). Reason: %s", 
                        error_string
                    )

            self.ib.errorEvent += on_error

            self.subscriptions = SubscriptionManager(self.ib)
            self.portfolio = PortfolioManager(self.ib, self.subscriptions)
            self.volatility = VolatilityManager(
                self.symbol, self.ib, self.subscriptions, self.risk_free_rate, self.div_yield
            )
            logger.info("Successfully connected to IBKR and initialized subsystems.")
        except Exception:
            logger.exception("CRITICAL: Failed to establish connection to IBKR.")
            raise

    def initialize(self) -> None:
        """Subscribes to the underlying and fetches the day's opening price."""

        self.portfolio.refresh()

        stock = Stock(self.symbol, "SMART", "USD")
        self.ib.qualifyContracts(stock)

        try:
            bars = self.ib.reqHistoricalData(stock, "", "1 D", "1 day", "TRADES", True)
            if bars:
                self.open_price = float(bars[-1].open)
                self.open_date = bars[-1].date.strftime("%d-%m-%Y")
        except Exception:
            logger.exception("Failed to fetch today's opening bar for %s.", self.symbol)

        self.underlying = self.subscriptions.subscribe(stock, _UNDERLYING_OWNER)
        self.ib.sleep(_STARTUP_SETTLE_SEC)

    def run(self) -> None:
        """Runs the main polling loop: refresh portfolio, refresh radar, redraw terminal."""
        try:
            while True:
                self.ib.sleep(_LOOP_INTERVAL_SEC)
                self.portfolio.refresh()
                self.volatility.build_radar(self.underlying)

                last = self.underlying.last
                close = self.underlying.close
                current_price = float(last if last and not np.isnan(last) else close)

                self.ui.render(
                    current_price=current_price,
                    open_price=self.open_price,
                    open_date=self.open_date,
                    expiries=self.volatility.expiries,
                    radar_data=self.volatility.radar_data,
                    net_liquidation=self.portfolio.net_liquidation,
                    available_funds=self.portfolio.available_funds,
                    excess_liquidity=self.portfolio.excess_liquidity,
                    total_subscriptions=self.subscriptions.total_subscriptions,
                )

                self.visualizer.update(
                    spot=current_price,
                    slices=self.volatility.surface_slices,
                )
        except KeyboardInterrupt:
            logger.info("Shutting down terminal safely...")
            self.visualizer.close()
            self.ib.disconnect()
        except Exception:
            logger.exception("Unhandled error in main loop.")
            self.ib.sleep(1)

    def start(self) -> None:
        """Connects, initializes, and runs the bot end to end."""
        self.connect()
        self.initialize()
        self.run()