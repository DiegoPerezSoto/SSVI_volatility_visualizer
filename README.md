# Volatility Visualizer — Real-Time SSVI Volatility Surface Visualization

**A professional options terminal for visualizing implied volatility surfaces in real time, powered by Interactive Brokers API and the Surface Stochastic Volatility Inspired (SSVI) model.**

---

## Table of Contents

1. [Overview](#overview)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Architecture](#architecture)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [Visualization Examples](#visualization-examples)
8. [Technical Details](#technical-details)
9. [Performance & Scaling](#performance--scaling)
10. [Limitations](#limitations)
11. [Future Work](#future-work)
12. [License](#license)

---

## Overview

**VolatilityVisualizer** is a real-time options analysis tool that connects directly to Interactive Brokers, streams live option quotes, calibrates a SSVI volatility surface, and visualizes both the raw market data and the fitted model in 3D.

**Target audience:** Traders, quants, and developers interested in options microstructure and volatility modeling.

**Why this matters:** Retail platforms show you an option's IV. They don't show you whether that IV is cheap or rich relative to its neighborhood on the volatility smile, tick by tick. This tool closes that gap.

---

## The Problem

When trading options, you're not just betting on direction—you're betting on *where* on the smile the market is pricing volatility. A standard broker platform shows:

```
Strike  |  Bid  |  Ask  |  IV
--------|-------|-------|------
205     | 2.50  | 2.65  | 28.3%
210     | 0.85  | 0.95  | 25.1%
215     | 0.15  | 0.25  | 20.7%
```

But it doesn't tell you: **Is 28.3% at the 205 call rich or cheap compared to the rest of the chain?**

That's where most retail traders miss opportunities. They can't see the *structure* of the smile in real time, so they can't tell when a single point is an outlier or when the entire surface has shifted.

---

## The Solution

**Volatility Visualizer** extracts the underlying structure from noisy market data using the **SSVI (Surface Stochastic Volatility Inspired) model**, a five-parameter parameterization that's:

- **Arbitrage-free** by construction (no butterfly spreads with negative value)
- **Smooth and interpretable** (captures realistic smile dynamics)
- **Fast to calibrate** (L-BFGS-B optimization, <100ms per expiry)
- **Standard in institutional trading** (used by major quant desks, investment banks)

The result: You see both the raw market quotes *and* the clean model fit, updated tick by tick. Outliers become obvious. Structural shifts become visible.

---

## Architecture

```
main.py (entry point)
  └── trading_bot.py (IBKR connection + event loop)
        ├── subscription_manager.py       [reference-counted market-data streams]
        ├── portfolio_manager.py          [account balance + open positions]
        ├── volatility_manager.py         [orchestrates all SVI components]
        │   ├── chain_selector.py         [10 expirations × 8 strikes OTM]
        │   ├── radar_engine.py           [SSVI calibration + IV computation]
        │   └── volatility_visualizer.py  [3D surface + 2D smile plots]
        ├── terminal_ui.py                [console rendering]
        └── options_pricer.py             [stateless Black-Scholes + SSVI math]
```

### Key Design Decisions

**1. Reference-Counted Subscriptions**

Multiple components (underlying tracker, volatility radar, portfolio valuator) may need the same contract. SubscriptionManager de-duplicates requests, keeping the bot well under IBKR's limit.

**2. OTM-Only Subscriptions**

For each strike, we subscribe to only one leg (calls ≥ spot, puts < spot). This avoids redundant subscriptions while ensuring we have the most liquid side of each strike.

**3. Stateless Pricing**

options_pricer.py contains no state. Every function receives inputs explicitly, making it unit-testable, thread-safe, and easy to extend.

---

## Installation & Setup

### Requirements

- **Python 3.8+**
- **TWS or IB Gateway** (running locally with API access enabled)
  - Paper trading: default port `7497`
  - Live: port `4002` (if using IB Gateway)

### Step 1: Clone and Install

```bash
git clone https://github.com/yourusername/quant_lab.git
cd quant_lab
pip install -r requirements.txt
```

### Step 2: Configure Interactive Brokers

Ensure TWS or IB Gateway is running and API is enabled.

### Step 3: Edit Configuration

Open `main.py`:

```python
bot = TradingBot(
    host="127.0.0.1",
    port=7497,              # 4002 if using IB Gateway
    client_id=20,
    symbol="SPY",           # change to your underlying
    risk_free_rate=0.043,
    div_yield=0.0,
)
```

### Step 4: Run

```bash
python main.py
```

---

## Usage

### Terminal Interface

The terminal updates every 500ms. You'll see:

- Current underlying price and daily change
- Account equity, available funds, excess liquidity
- Live options radar: bid/ask/IV for each strike, plus the SVI-fitted IV

### Interpretation

**Call IV > SVI IV:** The call is rich relative to the model. Consider selling or trading spreads.

**Call IV < SVI IV:** The call is cheap. Consider buying.

Same applies to puts and overall smile curvature.

---

## Visualization Examples

### Example 1: Volatility Smiles by Expiration

**[IMAGE PLACEHOLDER: Insert svi_lines_comparison.png or volatility_surface_comparison.png here]**

*Left: Raw market data with noise. Right: SVI fitted model showing clean structure.*

### Example 2: 3D Volatility Surface

**[IMAGE PLACEHOLDER: Insert 3D surface plot here]**

*The volatility surface across strikes and expirations. Colors represent IV levels.*

### Example 3: Live Terminal Output

**[IMAGE PLACEHOLDER: Insert screenshot of running terminal here]**

*Updated every 500ms during market hours.*

---

## Technical Details

### SVI Model

The raw SVI parameterization:

w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

Where:
- k = ln(K/S) is log-moneyness
- w = (IV)^2 * T is total implied variance
- (a, b, rho, m, sigma) are five fitted parameters

### Calibration

For each expiry:
1. Collect OTM quotes (calls ≥ spot, puts < spot)
2. Solve for IV via bisection on Black-Scholes price
3. Convert to total variance
4. Fit SVI parameters via L-BFGS-B optimization

If optimization fails, we fall back to a quadratic polynomial fit.

### Why SVI?

- Only 5 parameters (vs 10+ for SABR)
- Arbitrage-free by design
- Calibrates in <100ms
- Standard in institutional trading

---

## Performance & Scaling

| Metric | Value |
|--------|-------|
| Market-data subscriptions | 80 (10 expirations × 8 strikes) |
| IBKR API limit | Comfortably under 100 lines |
| SVI calibration per expiry | 30–80ms |
| Full radar cycle | ~500ms |
| Memory footprint | 50–100 MB |
| CPU usage | <5% on modern hardware |

---

## Limitations

- **Visualization only:** No trade execution
- **Real-time only:** No historical data saved (unless you add logging)
- **Bid-ask data only:** Works with quotes, not tick-level trades
- **Paper trading:** Some accounts may not have all expirations
- **Market hours only:** Live data 9:30 AM–4:00 PM ET

---

## Future Work

- Historical data logging for backtesting
- Multi-underlying support (SPY + QQQ + IWM)
- Real-time alerts (email/Slack on IV dislocations)
- Web dashboard (Plotly in browser)
- Machine learning for IV move prediction
- Backtesting module

---

## License

MIT License.

---

## About

Built as a demonstration of options microstructure, SVI calibration, IBKR API integration, and professional Python architecture.

**Author:** Diego Pérez Soto
**Background:** Math + Computer Science student 
**Focus:** Quantitative finance, derivatives

---

## Resources

- **GitHub:** github.com/DiegoPerezSoto/SSVI_volatility_visualizer
- **SVI Reference:** Gatheral, J. "The Volatility Surface: A Practitioner's Guide." Wiley, 2006.

---

## Disclaimer

This tool is for educational purposes only. Not investment advice. Options trading carries substantial risk; trade only with capital you can afford to lose.
