"""Entry point for QuantLab SVI Volatility Radar.

Usage:
    python main.py

Requires TWS or IB Gateway running locally with API access enabled.
Default ports: 7497 (TWS paper trading), 4002 (IB Gateway live).
"""

import logging
import multiprocessing

import nest_asyncio
import pandas as pd

from trading_bot import TradingBot
from volatility_visualizer import run_visualizer_process


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
    data_queue = multiprocessing.Queue()

    visualizer_process = multiprocessing.Process(target = run_visualizer_process, args = (data_queue,), name = "Visualizer", daemon = False)
    visualizer_process.start()
    logging.info("Dedicacated visualizer process succesfully started")
    bot = TradingBot(
            host="127.0.0.1",
            port=4002,
            client_id=20,
            symbol="NVDA",
            risk_free_rate=0.0375,
            div_yield=0.0,
        )

    bot.visualizer_queue = data_queue

    try:
        bot.start()
    finally:
        logging.info("Awaiting visualizer process termination...")
        visualizer_process.join(timeout=3.0)
        logging.info("All engine components shut down successfully.")


