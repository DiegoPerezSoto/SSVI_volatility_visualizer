"""Entry point for QuantLab SVI Volatility Radar.

Usage:
    python main.py

Requires TWS or IB Gateway running locally with API access enabled.
Default ports: 7497 (TWS paper trading), 4002 (IB Gateway live).
"""

import logging

import nest_asyncio
import pandas as pd

from trading_bot import TradingBot

# The terminal screen is cleared ~10x/second by TerminalUI, so anything sent
# only to the console is wiped before you can read it. Route logs to a file
# as well so `tail -f quantlab_debug.log` (or PowerShell equivalent) shows
# live, non-disappearing output of what's happening.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("quantlab_debug.log"),
        logging.StreamHandler(),
    ],
)

nest_asyncio.apply()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.colheader_justify", "center")

if __name__ == "__main__":
    bot = TradingBot(
        host="127.0.0.1",
        port=4002,
        client_id=20,
        symbol="META",
        risk_free_rate=0.043,
        div_yield=0.0,
    )
    bot.start()