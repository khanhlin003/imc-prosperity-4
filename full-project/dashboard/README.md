## Dashboard (Streamlit + Plotly)

Quick interactive viewer for IMC Prosperity backtester-style data (e.g. `data/round1/prices_round_1_day_*.csv` and `trades_round_1_day_*.csv`).

### Setup

On macOS with Homebrew Python (PEP 668), install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r dashboard/requirements.txt
python -m pip install -e backtester
```

### Run

From the repo root:

```bash
./.venv/bin/streamlit run dashboard/app.py
```

### What it shows

- Order book top-3 bid/ask levels over time (point size ~ volume)
- Mid price line
- Trades overlay (point size ~ quantity)
- Optional normalization by mid price (\(p_t - mid_t\))

