
# Intraday Dual SMA Backtesting & Statistical Analysis for NIFTY 50
<img width="2500" height="1800" alt="dual_sma_dash_board" src="https://github.com/user-attachments/assets/5e5c7009-33f0-4e2e-b4fd-1892b67a14ea" />

A comprehensive and extensible Python project to backtest single and dual Simple Moving Average (SMA) crossover strategies on 10 years of minute-wise NIFTY 50 data, with advanced statistics, risk/return metrics, rich visualization, and professional reporting.

---

## 📌 Project Highlights

- **End-to-End Backtesting**: Single and dual SMA strategies with configurable parameters.
- **Dataset**: 2015–2025, 932,946 rows of 1-min OHLCV bars for NIFTY 50 (India’s benchmark index).
- **Advanced Feature Engineering**: Time, price, volatility, and technical features for deeper analysis.
- **Trade Simulation**: Realistic execution with transaction cost, no lookahead bias, and precise trade accounting.
- **Statistical Deep-Dive**: Skew, kurtosis, tails, win/loss streaks, regime analysis, and more.
- **Dashboards**: Professional, multi-panel matplotlib/seaborn plots summarizing all strategic and statistical dimensions.
- **Reproducible & Modular**: Fully-commented, extensible, and production-friendly codebase.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Usage](#installation--usage)
3. [Methodology](#methodology)
4. [Key Results & Interpretations](#key-results--interpretations)
5. [Statistical Analysis & Insights](#statistical-analysis--insights)
6. [Dashboards & Visuals](#dashboards--visuals)
7. [Limitations & Next Steps](#limitations--next-steps)
8. [How to Extend](#how-to-extend)
9. [License](#license)
10. [Author / Contact](#author--contact)

---

## Introduction

This repository contains Python code, workflow, and results for robustly benchmarking single and dual SMA crossovers as intraday trend-following strategies on NIFTY 50. The project evaluates not just return but also risk, trade behavior, and statistical robustness, providing a plug-and-play framework for further research or production deployment.

---

## Installation & Usage

### 1. Requirements

- Python 3.7+ (Anaconda recommended)
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

### 2. Dataset

- **File:** `NIFTY 50_minute_data.csv`
- **Columns:** DateTime (index), Open, High, Low, Close, Volume

### 3. How to Run

```
git clone https://github.com/shubh123a3/Intraday-Dual-SMA-Backtesting-Statistical-Analysis-for-NIFTY-50.git
cd Intraday-Dual-SMA-Backtesting-Statistical-Analysis-for-NIFTY-50
python app.py
```
- All main analytics run from `app.py`. Results, tables, and plots are saved in the working directory.

---

## Methodology

### Data Audit
- Full decade, minute-level NIFTY 50 OHLCV. **No missing data.**

### Feature Engineering
- **SMAs**: Rolling means for periods 5, 10, 20, 50, 100, 200; all dual SMA pairs (fast  `SMA(N)`
- Flat otherwise

#### Dual SMA
- Long if `SMA(fast)` > `SMA(slow)`
- Flat otherwise

- All entries/exits strictly at the next bar to avoid look-ahead.
- Transaction cost: 0.015% per round-trip per trade.

### Backtest Metrics

- Total and annualized return
- Annualized volatility
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown
- Win rate, trade count, profit factor
- Tail and percentile statistics
- Monthly regime analysis

---

## Key Results & Interpretations

### Data Stats

| Attribute  | Value          |
|------------|---------------|
| Bars       | 932,946       |
| Mean Close | 13,624        |
| Coverage   | 2015–2025     |
| Gaps/Missing | 0           |

### Single SMA Results

| SMA       | Total Return | Sharpe Ratio | Max Drawdown | Win Rate (%) | Trades  |
|-----------|--------------|--------------|--------------|-------------|--------|
| 5         | −82.0%       | −3.18        | −99.7%       | 6.4         | 25,558 |
| 10        | −64.5%       | −2.43        | −99.4%       | 8.8         | 23,001 |
| 20        | −8.6%        | −0.43        | −92.1%       | 10.8        | 18,012 |
| 50        | +98.0%       | 0.57         | −44.9%       | 11.8        | 13,554 |
| 100       | −40.0%       | −0.64        | −75.7%       | 9.4         | 10,112 |
| 200       | −0.5%        | −2.17        | −70.1%       | 9.9         | 8,006  |

### Dual SMA Crossover Results (Top 5)

| Strategy  | Sharpe | Total Return | Calmar | Max DD    | Win Rate | Trades  |
|-----------|--------|-------------|--------|-----------|----------|--------|
| 50/200    | 1.05   | 191.5%      | 10.68  | −17.9%    | 12.1%    | 5,480  |
| 20/200    | 0.60   | 80.4%       | 3.82   | −21.0%    | 11.0%    | 7,721  |
| 50/100    | 0.10   | 4.4%        | 0.13   | −35.0%    | 11.5%    | 10,251 |
| 10/200    | −0.01  | −6.4%       | −0.15  | −41.8%    | 8.8%     | 10,465 |
| 20/100    | −0.16  | −20.2%      | −0.47  | −42.9%    | 11.0%    | 12,469 |

**Best strategy:**  
**50/200** — Sharpe 1.05, Total Return 191.5%, Max Drawdown −17.9%.

---

## Statistical Analysis & Insights

### Detailed Stats for Best Dual SMA (50/200)

- **Annualized Return**: 11.35%
- **Volatility**: 10.75%
- **Sortino Ratio**: 0.86
- **Win Rate**: 12.1%
- **Mean Trade Return**: −0.018%
- **Skewness**: −20.38 (pronounced left tail)
- **Kurtosis**: 784.3 (fat tails; rare large moves drive edge)
- **Tail Risk**: Top 5% of trades have strong negative and positive impact
- **Drawdown**: Max −17.9%
- **Months Positive**: 63% (best month: +9.35%, worst: −4.60%)

### Comparative Analysis

| Metric            | Best Single SMA (50) | Best Dual SMA (50/200) | Improvement |
|-------------------|---------------------|------------------------|-------------|
| Sharpe Ratio      | 0.57                | 1.05                   | +84%        |
| Total Return      | 98.0%               | 191.5%                 | +95%        |
| Max Drawdown      | −44.9%              | −17.9%                 | −60% risk   |

**Dual crossovers sharply outperform single SMAs in return and risk-adjusted terms.**

### Market Regime Insights

- Outperformance is concentrated in high-volatility periods (e.g., COVID crash, 2022 spikes)
- At least 63% of months are profitable, with strong positive autocorrelation during trending quarters

---

## Dashboards & Visuals

**Auto-generated multi-panel dashboards:**

- **Single SMA Comparison:** Cumulative return, drawdown, rolling Sharpe, distribution/inferential plots
- **Dual SMA Performance Heatmaps:** Sharpe, win rate, total returns for all combinations
- **Trade Diagnostics:** Distribution histograms, boxplots, win/loss streaks
- **Risk & Return Visuals:** Volatility-return scatters, drawdown profiles, monthly regime heatmaps

You can find sample output images and notebooks in the repo, such as:
- `single_ema_dash.jpg`
- `dual_sma_dash_board.jpg`
- `Nifity-50_data_stats.jpg`

---

## Limitations & Next Steps

- **Volume** data is zeroed in the current dataset, limiting liquidity filtering.
- **Low hit rate** (win %): Alpha primarily from tail events; not “smooth.”
- **Practicality:** Slippage and trading frictions in live execution could erode returns.
- **Further research suggestions**:
  - Add volatility regime filters (ATR, realized vol)
  - Try more sophisticated portfolio overlays
  - Test on additional Indian indices (BANKNIFTY, SENSEX)
  - Integrate tick-level spreads and cost models

---

## How to Extend

- Add more moving average windows or new indicator features in `app.py`.
- Modular design allows swapping in different signal logic or ML-based classifiers.
- Export results to CSV/Excel; integrate with visualization libraries (Plotly, Altair).
- Adapt for use with real-time data or alternative asset universes.

---

## License

This project is open-sourced under the MIT License—see [LICENSE](LICENSE) for details.

---

## Author / Contact

**Shubh Shrishrimal.**  
For questions or collaborations, open an [issue](https://github.com/shubh123a3/Intraday-Dual-SMA-Backtesting-Statistical-Analysis-for-NIFTY-50/issues) or connect via LinkedIn.

---

## Acknowledgement

Market data used for research and educational purposes. No investment advice is given; all trading involves risk.

---

## Repository Structure

```
├── app.py                   # Main analysis pipeline
├── NIFTY 50_minute_data.csv # Minute-wise index data (not distributed in repo)
├── images/                  # Key analysis Dashboards and charts
├── requirements.txt         # Python library dependencies
└── README.md                # This file
```

---

**Sample Output Excerpt:**
```
Best Dual SMA Strategy: 50/200
Performance: Sharpe 1.05, Total Return 191.5%, Max Drawdown -17.9%
Statistical Features: Skew -20.4, Kurtosis 784.3, Top 5% tail trades drive edge
Months Profitable: 63%
```

---
