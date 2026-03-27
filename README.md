# Reinforcement Learning Trading Bot for BTC/USDT

**PPO Agent with Custom Gymnasium Environment**

This project implements a reinforcement learning trading bot for Bitcoin (BTC/USDT) using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3. The bot learns to open and close long/short positions on 15-minute timeframe data, incorporating technical indicators and market session logic.

**Project Status:** Work in Progress (WIP)

---

## Project Overview

The goal of this project is to develop an autonomous trading agent capable of making sequential trading decisions in a realistic environment. The agent operates on historical BTC/USDT 15-minute candles and uses a custom Gymnasium environment to simulate trading with leverage, risk management, and contextual market information.

Key features include:
- Custom `TradingEnv` compatible with the Gymnasium standard
- Rich observation space: current OHLC + ATR + Order Blocks + Fair Value Gaps + Kill Zone indicator + 96-step historical close prices (1-day context)
- Action space: Close position, Open Long, Open Short, Wait
- Entry logic restricted to Kill Zones with confirmation from Order Blocks or Fair Value Gaps
- ATR-based Stop Loss and Take Profit risk management
- Shaped reward function with strong signals for Take Profit and penalties for Stop Loss
- Comprehensive transaction logging and performance analysis (win rate, max drawdown, profit/loss)

---

## Data Preparation

Historical market data is downloaded directly from the **Binance API** and enriched with technical features.

The script **`dane.py`** handles the entire data pipeline:
- Fetches raw 15-minute BTC/USDT klines from Binance (from March 2020 to March 2025)
- Calculates ATR (Average True Range)
- Detects Order Blocks (bullish and bearish)
- Identifies Fair Value Gaps (FVG) with direction
- Adds Kill Zone sessions (high-volatility trading windows)
- Saves the processed dataset as `btc_usd_2020_2025_15mv2.csv`

To prepare the dataset, run:
```bash
python dane.py
