from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from importlib import import_module, reload

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_BT_RESOURCES_DIR = ROOT / "data"

# Streamlit keeps one Python process; a stale `prosperity3bt` from site-packages can stay in
# `sys.modules` forever. We always prefer repo `backtester/` and drop cached imports before use.
_REPO_BACKTESTER = (ROOT / "backtester").resolve()


def _purge_and_use_repo_prosperity3bt() -> None:
    """Put `backtester/` first on sys.path and remove any loaded `prosperity3bt` so the next import is from this repo."""
    if not _REPO_BACKTESTER.is_dir():
        return
    rp = str(_REPO_BACKTESTER)
    if rp in sys.path:
        sys.path.remove(rp)
    sys.path.insert(0, rp)
    for name in list(sys.modules.keys()):
        if name == "prosperity3bt" or name.startswith("prosperity3bt."):
            del sys.modules[name]


_purge_and_use_repo_prosperity3bt()


def _dashboard_default_data_dir() -> Path:
    """Match `prosperity3bt` CLI resolution (`data/` next to `backtester/`, else `./data`)."""
    try:
        from prosperity3bt.file_reader import default_data_root

        dr = default_data_root()
        if dr is not None:
            return dr
    except Exception:
        pass
    return DEFAULT_BT_RESOURCES_DIR

def st_chart(fig) -> None:
    """Streamlit compatibility: width='stretch' (new) vs use_container_width (old)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def st_table(df: pd.DataFrame) -> None:
    """Streamlit compatibility: width='stretch' (new) vs use_container_width (old)."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


@dataclass(frozen=True)
class DayFiles:
    day: int
    prices_path: Path
    trades_path: Path


def _available_rounds(data_root: Path) -> list[int]:
    if not data_root.exists():
        return []
    rounds = []
    for d in data_root.glob("round*"):
        if not d.is_dir():
            continue
        try:
            rounds.append(int(d.name.replace("round", "")))
        except Exception:
            continue
    return sorted(set(rounds))

def _parse_data_path(data_path: Path) -> tuple[Path, int]:
    """
    Accept either:
    - a root containing round folders (e.g. data/ with data/round1, data/round2, ...)
    - a direct round folder path (e.g. .../resources/round2)
    Returns (round_dir, round_num).
    """
    name = data_path.name
    if name.startswith("round"):
        try:
            r = int(name.replace("round", ""))
            return data_path, r
        except Exception:
            pass
    raise ValueError("Not a round folder")


def _find_round_days(data_dir: Path, round_num: int) -> list[DayFiles]:
    if not data_dir.exists():
        return []

    days: dict[int, dict[str, Path]] = {}
    for p in sorted(data_dir.glob(f"prices_round_{round_num}_day_*.csv")):
        # filenames look like: prices_round_1_day_-2.csv
        try:
            day = int(p.stem.split("day_")[-1])
        except Exception:
            continue
        days.setdefault(day, {})["prices"] = p

    for p in sorted(data_dir.glob(f"trades_round_{round_num}_day_*.csv")):
        try:
            day = int(p.stem.split("day_")[-1])
        except Exception:
            continue
        days.setdefault(day, {})["trades"] = p

    out: list[DayFiles] = []
    for day, files in days.items():
        if "prices" in files and "trades" in files:
            out.append(DayFiles(day=day, prices_path=files["prices"], trades_path=files["trades"]))

    return sorted(out, key=lambda x: x.day)


@st.cache_data(show_spinner=False)
def load_prices(prices_path: str) -> pd.DataFrame:
    df = pd.read_csv(prices_path, sep=";")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "product"]).sort_values(["product", "timestamp"])
    df["timestamp"] = df["timestamp"].astype(int)

    # Some datasets encode "no book" as NaNs for best bid/ask, while keeping mid_price at 0.
    # Treat those mids as missing to avoid plots/normalization being dragged to zero.
    for col in ("bid_price_1", "ask_price_1", "mid_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"bid_price_1", "ask_price_1", "mid_price"}.issubset(df.columns):
        no_book = df["bid_price_1"].isna() & df["ask_price_1"].isna()
        df.loc[no_book & (df["mid_price"] == 0), "mid_price"] = np.nan

    return df


@st.cache_data(show_spinner=False)
def load_trades(trades_path: str) -> pd.DataFrame:
    df = pd.read_csv(trades_path, sep=";")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "symbol", "price", "quantity"]).sort_values(["symbol", "timestamp"])
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def _melt_book_levels(prices: pd.DataFrame, side: str) -> pd.DataFrame:
    assert side in ("bid", "ask")
    price_cols = [f"{side}_price_{i}" for i in (1, 2, 3)]
    vol_cols = [f"{side}_volume_{i}" for i in (1, 2, 3)]

    cols = ["timestamp"] + [c for c in price_cols + vol_cols if c in prices.columns]
    df = prices[cols].copy()

    price_long = df.melt(id_vars=["timestamp"], value_vars=[c for c in price_cols if c in df.columns], var_name="level", value_name="price")
    vol_long = df.melt(id_vars=["timestamp"], value_vars=[c for c in vol_cols if c in df.columns], var_name="level", value_name="volume")

    # map bid_price_1 -> 1 etc.
    price_long["level_n"] = price_long["level"].str.extract(r"_(\d+)$").astype(float)
    vol_long["level_n"] = vol_long["level"].str.extract(r"_(\d+)$").astype(float)

    out = price_long.merge(vol_long[["timestamp", "level_n", "volume"]], on=["timestamp", "level_n"], how="left")
    out = out.dropna(subset=["price"])
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
    out["side"] = side
    out["book_level"] = out["level_n"].astype(int)
    # Keep only the derived numeric level; drop helper columns
    out = out.drop(columns=["level_n", "level"])
    return out


def _add_wall_mid(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute Frankfurt-style Wall Mid from available top-of-book levels."""
    bid_cols = [c for c in prices.columns if c.startswith("bid_price_")]
    ask_cols = [c for c in prices.columns if c.startswith("ask_price_")]
    if not bid_cols or not ask_cols:
        return prices

    out = prices.copy()
    for c in bid_cols + ask_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["bid_wall"] = out[bid_cols].min(axis=1, skipna=True)
    out["ask_wall"] = out[ask_cols].max(axis=1, skipna=True)
    out["wall_mid"] = (out["bid_wall"] + out["ask_wall"]) / 2.0

    # If there's no visible book, avoid spurious zeros
    no_book = out["bid_wall"].isna() & out["ask_wall"].isna()
    out.loc[no_book, "wall_mid"] = np.nan
    return out


def _normalize_by_baseline(prices: pd.DataFrame, trades: pd.DataFrame, baseline_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize y-values by a baseline at same/previous timestamp:
      y' = y - baseline(t)
    """
    if baseline_col not in prices.columns:
        return prices, trades

    base = prices[["timestamp", baseline_col]].copy().sort_values("timestamp")
    base[baseline_col] = pd.to_numeric(base[baseline_col], errors="coerce")

    prices_n = prices.copy()
    for col in [c for c in prices_n.columns if c.startswith("bid_price_") or c.startswith("ask_price_") or c in ("mid_price", "wall_mid", "bid_wall", "ask_wall")]:
        if col not in prices_n.columns:
            continue
        prices_n[col] = pd.to_numeric(prices_n[col], errors="coerce")
        prices_n[col] = prices_n[col] - prices_n[baseline_col]

    trades_n = trades.copy().sort_values("timestamp")
    if not trades_n.empty and "price" in trades_n.columns:
        trades_n["price"] = pd.to_numeric(trades_n["price"], errors="coerce")
        trades_n = pd.merge_asof(trades_n, base, on="timestamp", direction="backward")
        trades_n["price"] = trades_n["price"] - trades_n[baseline_col]
    return prices_n, trades_n


def make_figure(
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    normalize: bool,
    levels: tuple[int, ...],
    auto_y_zoom: bool,
    flip_y: bool,
    trade_marker_symbol: str = "triangle-up",
    trade_trace_name: str = "Trades",
) -> go.Figure:
    # Build wall-mid for visualization + optional normalization
    prices = _add_wall_mid(prices)

    if normalize != "None":
        baseline_col = "mid_price" if normalize == "Mid" else "wall_mid"
        prices, trades = _normalize_by_baseline(prices, trades, baseline_col=baseline_col)

    bids = _melt_book_levels(prices, "bid")
    asks = _melt_book_levels(prices, "ask")
    if levels:
        bids = bids[bids["book_level"].isin(levels)]
        asks = asks[asks["book_level"].isin(levels)]

    fig = go.Figure()

    def _size(vol: pd.Series) -> np.ndarray:
        v = pd.to_numeric(vol, errors="coerce").fillna(0).to_numpy()
        # keep readable: sqrt scaling + floor/ceil bounds
        s = 4.0 + 2.5 * np.sqrt(np.maximum(v, 0))
        return np.clip(s, 4, 18)

    if not bids.empty:
        fig.add_trace(
            go.Scattergl(
                x=bids["timestamp"],
                y=bids["price"],
                mode="markers",
                name=f"Bids (L{min(levels)}–L{max(levels)})" if levels else "Bids",
                marker=dict(color="rgba(30, 136, 229, 0.75)", size=_size(bids["volume"])),
                hovertemplate="t=%{x}<br>bid=%{y}<br>vol=%{customdata}<extra></extra>",
                customdata=bids["volume"],
            )
        )

    if not asks.empty:
        fig.add_trace(
            go.Scattergl(
                x=asks["timestamp"],
                y=asks["price"],
                mode="markers",
                name=f"Asks (L{min(levels)}–L{max(levels)})" if levels else "Asks",
                marker=dict(color="rgba(229, 57, 53, 0.75)", size=_size(asks["volume"])),
                hovertemplate="t=%{x}<br>ask=%{y}<br>vol=%{customdata}<extra></extra>",
                customdata=asks["volume"],
            )
        )

    if "mid_price" in prices.columns:
        fig.add_trace(
            go.Scatter(
                x=prices["timestamp"],
                y=prices["mid_price"],
                mode="lines",
                name="Mid",
                line=dict(color="rgba(60, 60, 60, 0.65)", width=2),
                hovertemplate="t=%{x}<br>mid=%{y}<extra></extra>",
            )
        )

    if not trades.empty:
        qty_col = "quantity" if "quantity" in trades.columns else "signed_qty"
        qty = pd.to_numeric(trades[qty_col], errors="coerce").fillna(0).abs()
        fig.add_trace(
            go.Scattergl(
                x=trades["timestamp"],
                y=trades["price"],
                mode="markers",
                name=trade_trace_name,
                marker=dict(
                    color="rgba(0, 0, 0, 0.95)",
                    # Make trades visually dominant: bigger triangle + stronger scaling
                    size=np.clip(12 + 3.2 * np.sqrt(np.maximum(qty.to_numpy(), 0)), 12, 42),
                    line=dict(color="rgba(0,0,0,0.8)", width=1),
                    symbol=trade_marker_symbol,
                ),
                hovertemplate="t=%{x}<br>px=%{y}<br>qty=%{customdata}<extra></extra>",
                customdata=trades[qty_col],
            )
        )

    title = "Order book + trades" + (f" (normalized by {normalize.lower()})" if normalize != "None" else "")
    fig.update_layout(
        title=title,
        xaxis_title="timestamp",
        yaxis_title="price" + (f" - {('mid_price' if normalize == 'Mid' else 'wall_mid')}" if normalize != "None" else ""),
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", rangeslider=dict(visible=True))
    # Y-zoom: auto-focus on bid/ask band in the currently selected window
    if auto_y_zoom:
        y_series = []
        if not bids.empty:
            y_series.append(pd.to_numeric(bids["price"], errors="coerce"))
        if not asks.empty:
            y_series.append(pd.to_numeric(asks["price"], errors="coerce"))
        if len(y_series) > 0:
            y_all = pd.concat(y_series).dropna()
            if len(y_all) > 0:
                y_low = float(y_all.min())
                y_high = float(y_all.max())
                span = max(1.0, y_high - y_low)
                pad = max(2.0, 0.08 * span)

                if flip_y:
                    fig.update_yaxes(range=[y_high + pad, y_low - pad], autorange=False)
                else:
                    fig.update_yaxes(range=[y_low - pad, y_high + pad], autorange=False)

    if flip_y and not auto_y_zoom:
        fig.update_yaxes(autorange="reversed")

    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    return fig


def _algorithms_in_repo(root: Path) -> list[Path]:
    candidates = []
    for p in sorted(root.glob("*.py")):
        if p.name.startswith("."):
            continue
        if p.name == "__init__.py":
            continue
        candidates.append(p)
    return candidates


def _available_round_days(resources_dir: Path) -> dict[int, list[int]]:
    out: dict[int, set[int]] = {}
    if not resources_dir.exists():
        return {}
    for round_dir in resources_dir.glob("round*"):
        try:
            r = int(round_dir.name.replace("round", ""))
        except Exception:
            continue
        for p in round_dir.glob("prices_round_*_day_*.csv"):
            stem = p.stem
            # prices_round_1_day_-2
            try:
                day = int(stem.split("day_")[-1])
            except Exception:
                continue
            out.setdefault(r, set()).add(day)
    return {r: sorted(list(days)) for r, days in sorted(out.items(), key=lambda x: x[0])}


@st.cache_data(show_spinner=False)
def _run_backtest_cached(
    algorithm_path_str: str,
    resources_dir_str: str,
    round_days: tuple[tuple[int, int], ...],
    match_trades: str,
    merge_pnl: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run prosperity3bt in-process and return:
    - activities dataframe (includes profit_and_loss per product over time)
    - trades dataframe (submission-only trades, signed qty)
    - sandbox dataframe (timestamp, lambda_log) for trader stdout capture
    - market_trades dataframe (timestamp, symbol, price, quantity) from `data/*`
    """
    _purge_and_use_repo_prosperity3bt()

    def _import_backtester():
        from prosperity3bt.file_reader import FileSystemReader
        from prosperity3bt.models import TradeMatchingMode  # pyright: ignore[reportMissingImports]
        from prosperity3bt.runner import run_backtest

        return FileSystemReader, TradeMatchingMode, run_backtest

    try:
        FileSystemReader, TradeMatchingMode, run_backtest = _import_backtester()
    except Exception as e:
        raise RuntimeError(
            "Could not import `prosperity3bt`.\n\n"
            "Fix options:\n"
            "- Run Streamlit from your venv: `./.venv/bin/streamlit run dashboard/app.py`\n"
            "- Or install into the interpreter running Streamlit: `./.venv/bin/python -m pip install -e backtester`\n"
        ) from e

    algorithm_path = Path(algorithm_path_str).resolve()
    resources_dir = Path(resources_dir_str).resolve()

    # Import algorithm module (same approach as prosperity3bt CLI)
    sys.path.append(str(algorithm_path.parent))
    trader_module = import_module(algorithm_path.stem)
    if not hasattr(trader_module, "Trader"):
        raise RuntimeError(f"{algorithm_path.name} does not expose a Trader class")

    file_reader = FileSystemReader(resources_dir)
    mode = TradeMatchingMode(match_trades)

    # Run each requested day, then merge like prosperity3bt does
    merged_activity_rows = []
    merged_trade_rows = []
    merged_sandbox_rows: list[list] = []
    merged_market_trade_rows: list[list] = []
    profit_offsets: dict[str, float] = {}
    timestamp_offset = 0

    for i, (round_num, day_num) in enumerate(round_days):
        reload(trader_module)
        result = run_backtest(
            trader_module.Trader(),
            file_reader,
            round_num,
            day_num,
            False,  # print_output
            mode,
            True,  # no_names (matches prosperity3bt CLI; unused in current read_day_data)
            False,  # show_progress_bar
        )

        # Convert activity logs to rows
        day_activity = [row.columns[:] for row in result.activity_logs]

        # Apply timestamp offset after first chunk (to make time continuous)
        if i > 0:
            timestamp_offset = (merged_activity_rows[-1][1] if merged_activity_rows else 0) + 100

        # If merging PnL, add last pnl per product as offset
        if merge_pnl and merged_activity_rows:
            last_ts = merged_activity_rows[-1][1]
            profit_offsets = {}
            for row in reversed(merged_activity_rows):
                if row[1] != last_ts:
                    break
                profit_offsets[row[2]] = float(row[-1])

        for row in day_activity:
            row[1] = int(row[1]) + int(timestamp_offset)
            if merge_pnl:
                row[-1] = float(row[-1]) + float(profit_offsets.get(row[2], 0.0))
            merged_activity_rows.append(row)

        # Market trades from data files (existing trades)
        try:
            market_path = resources_dir / f"round{round_num}" / f"trades_round_{round_num}_day_{day_num}.csv"
            if market_path.is_file():
                mt = pd.read_csv(market_path, sep=";")
                # canonicalize
                if {"timestamp", "buyer", "seller", "symbol", "price", "quantity"}.issubset(mt.columns):
                    mt = mt[["timestamp", "symbol", "price", "quantity", "buyer", "seller"]].copy()
                    mt["timestamp"] = pd.to_numeric(mt["timestamp"], errors="coerce")
                    mt = mt.dropna(subset=["timestamp", "symbol", "price", "quantity", "buyer", "seller"])
                    mt["timestamp"] = mt["timestamp"].astype(int) + int(timestamp_offset)
                    mt["price"] = pd.to_numeric(mt["price"], errors="coerce").astype(int)
                    mt["quantity"] = pd.to_numeric(mt["quantity"], errors="coerce").astype(int)
                    mt["buyer"] = mt["buyer"].astype(str)
                    mt["seller"] = mt["seller"].astype(str)
                    merged_market_trade_rows.extend(mt.values.tolist())
        except Exception:
            # If the file is missing or malformed, skip market trade overlay.
            pass

        # Trades: keep only submission trades with signed qty
        for tr in result.trades:
            t = tr.trade
            ts = int(t.timestamp) + int(timestamp_offset)
            signed_qty = 0
            if t.buyer == "SUBMISSION":
                signed_qty = int(t.quantity)
            elif t.seller == "SUBMISSION":
                signed_qty = -int(t.quantity)
            else:
                continue
            merged_trade_rows.append([ts, t.symbol, signed_qty, int(t.price)])

        for sb in result.sandbox_logs:
            tsb = int(sb.timestamp) + int(timestamp_offset)
            merged_sandbox_rows.append([tsb, sb.lambda_log or ""])

    activities = pd.DataFrame(
        merged_activity_rows,
        columns=[
            "day",
            "timestamp",
            "product",
            "bid_price_1",
            "bid_volume_1",
            "bid_price_2",
            "bid_volume_2",
            "bid_price_3",
            "bid_volume_3",
            "ask_price_1",
            "ask_volume_1",
            "ask_price_2",
            "ask_volume_2",
            "ask_price_3",
            "ask_volume_3",
            "mid_price",
            "profit_and_loss",
        ],
    )
    activities["timestamp"] = pd.to_numeric(activities["timestamp"], errors="coerce").astype(int)
    activities["profit_and_loss"] = pd.to_numeric(activities["profit_and_loss"], errors="coerce")

    trades = pd.DataFrame(merged_trade_rows, columns=["timestamp", "symbol", "signed_qty", "price"])
    if not trades.empty:
        trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="coerce").astype(int)
        trades["signed_qty"] = pd.to_numeric(trades["signed_qty"], errors="coerce").astype(int)
        trades["price"] = pd.to_numeric(trades["price"], errors="coerce").astype(int)

    sandbox = pd.DataFrame(merged_sandbox_rows, columns=["timestamp", "lambda_log"])
    if not sandbox.empty:
        sandbox["timestamp"] = pd.to_numeric(sandbox["timestamp"], errors="coerce").astype(int)

    market_trades = pd.DataFrame(merged_market_trade_rows, columns=["timestamp", "symbol", "price", "quantity", "buyer", "seller"])
    if not market_trades.empty:
        market_trades["timestamp"] = pd.to_numeric(market_trades["timestamp"], errors="coerce").astype(int)
        market_trades["price"] = pd.to_numeric(market_trades["price"], errors="coerce").astype(int)
        market_trades["quantity"] = pd.to_numeric(market_trades["quantity"], errors="coerce").astype(int)
        market_trades["buyer"] = market_trades["buyer"].astype(str)
        market_trades["seller"] = market_trades["seller"].astype(str)

    return activities, trades, sandbox, market_trades


def _final_pnl_by_product(activities: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    PnL at the final timestamp per product (same idea as `prosperity3bt` CLI day summary).
    Returns (table with columns product, profit_and_loss), total across products.
    """
    if activities.empty:
        return pd.DataFrame(columns=["product", "profit_and_loss"]), 0.0
    last_ts = int(activities["timestamp"].max())
    last = activities[activities["timestamp"] == last_ts]
    by_prod = last.groupby("product", as_index=False)["profit_and_loss"].first()
    by_prod = by_prod.sort_values("product").reset_index(drop=True)
    total = float(by_prod["profit_and_loss"].sum())
    return by_prod, total


def _pnl_y_range_for_display(y: pd.Series | np.ndarray, *, full_range: bool) -> tuple[float, float] | None:
    """
    Mark-to-market PnL often has rare huge spikes; a linear axis then hides the typical path.
    Default: zoom y to a percentile band so the bulk of the series is visible.
    full_range=True or degenerate data: return None so Plotly uses full autorange.
    """
    if full_range:
        return None
    s = pd.to_numeric(pd.Series(y), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if s.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.quantile(s, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    span = float(hi - lo)
    pad = max(1.0, span * 0.08)
    lo, hi = lo - pad, hi + pad
    if lo >= hi:
        mid = 0.5 * (lo + hi)
        return mid - 1.0, mid + 1.0
    return lo, hi


def _summary_with_trades(activities: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Final PnL per product plus submission trade counts and gross signed volume."""
    summary_df, total_pnl = _final_pnl_by_product(activities)
    if summary_df.empty:
        return summary_df, total_pnl
    if trades.empty:
        out = summary_df.copy()
        out["n_trades"] = 0
        out["gross_qty"] = 0
        return out, total_pnl

    g = trades.groupby("symbol", as_index=False).agg(
        n_trades=("timestamp", "count"),
        gross_qty=("signed_qty", lambda s: int(pd.Series(s).abs().sum())),
    )
    g = g.rename(columns={"symbol": "product"})
    out = summary_df.merge(g, on="product", how="left")
    out["n_trades"] = out["n_trades"].fillna(0).astype(int)
    out["gross_qty"] = out["gross_qty"].fillna(0).astype(int)
    return out, total_pnl


def _plot_all_products_pnl(activities: pd.DataFrame, *, pnl_full_y: bool) -> go.Figure:
    fig = go.Figure()
    all_pnl: list[pd.Series] = []
    for prod in sorted(activities["product"].unique()):
        s = activities[activities["product"] == prod].sort_values("timestamp")
        all_pnl.append(s["profit_and_loss"])
        n = len(s)
        fig.add_trace(
            go.Scatter(
                x=s["timestamp"],
                y=s["profit_and_loss"],
                mode="lines",
                name=prod,
                customdata=np.full(n, prod, dtype=object),
                hovertemplate="%{customdata}<br>t=%{x}<br>PnL=%{y}<extra></extra>",
            )
        )
    combined = pd.concat(all_pnl, ignore_index=True) if all_pnl else pd.Series(dtype=float)
    y_rng = _pnl_y_range_for_display(combined, full_range=pnl_full_y)
    fig.update_layout(
        title="PnL by product (mark-to-market)",
        height=440,
        margin=dict(l=40, r=24, t=50, b=40),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        # "x unified" can draw a red error overlay with some Plotly/Streamlit combos when traces differ a lot
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="timestamp")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title="profit_and_loss")
    if y_rng is not None:
        fig.update_yaxes(range=list(y_rng))
    return fig


def _plot_pnl_and_position(
    activities: pd.DataFrame, trades: pd.DataFrame, product: str, *, pnl_full_y: bool
) -> None:
    act = activities[activities["product"] == product].sort_values("timestamp").reset_index(drop=True)
    if act.empty:
        st.warning(f"No activity rows for {product}.")
        return

    # Position reconstruction from trades (merge_asof needs non-colliding column names)
    pos = pd.DataFrame({"timestamp": act["timestamp"]})
    tr = trades[trades["symbol"] == product].sort_values("timestamp") if not trades.empty else pd.DataFrame()
    if not tr.empty:
        tr_pos = tr.groupby("timestamp", as_index=False)["signed_qty"].sum()
        tr_pos["position"] = tr_pos["signed_qty"].cumsum()
        pos = pd.merge_asof(
            pos.sort_values("timestamp"),
            tr_pos[["timestamp", "position"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        pos["position"] = pos["position"].fillna(0).astype(int)
    else:
        pos["position"] = 0

    pnl_fig = go.Figure()
    pnl_fig.add_trace(
        go.Scatter(x=act["timestamp"], y=act["profit_and_loss"], mode="lines", name="PnL (mark-to-market)")
    )
    y_rng = _pnl_y_range_for_display(act["profit_and_loss"], full_range=pnl_full_y)
    pnl_fig.update_layout(height=260, margin=dict(l=30, r=20, t=35, b=25), title=f"{product}: PnL over time")
    pnl_fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    pnl_fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    if y_rng is not None:
        pnl_fig.update_yaxes(range=list(y_rng))

    pos_fig = go.Figure()
    pos_fig.add_trace(go.Scatter(x=pos["timestamp"], y=pos["position"], mode="lines", name="Position"))
    pos_fig.update_layout(height=260, margin=dict(l=30, r=20, t=35, b=25), title=f"{product}: Position over time")
    pos_fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    pos_fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    st_chart(pnl_fig)
    st_chart(pos_fig)


def _plot_backtest_book_and_trades(
    activities: pd.DataFrame,
    bot_trades: pd.DataFrame,
    market_trades: pd.DataFrame,
    product: str,
    *,
    normalize: str,
    levels: tuple[int, ...],
    auto_y_zoom: bool,
    flip_y: bool,
    window: tuple[int, int] | None,
    highlight_trader: str | None,
) -> None:
    prices = activities[activities["product"] == product].sort_values("timestamp").reset_index(drop=True)
    bot_trades = bot_trades[bot_trades["symbol"] == product].sort_values("timestamp").reset_index(drop=True)
    market_trades = market_trades[market_trades["symbol"] == product].sort_values("timestamp").reset_index(drop=True)

    if prices.empty:
        st.warning(f"No activity rows for {product}.")
        return

    if window is not None:
        t0, t1 = window
        prices = prices[(prices["timestamp"] >= t0) & (prices["timestamp"] <= t1)]
        bot_trades = bot_trades[(bot_trades["timestamp"] >= t0) & (bot_trades["timestamp"] <= t1)]
        market_trades = market_trades[(market_trades["timestamp"] >= t0) & (market_trades["timestamp"] <= t1)]

    # Normalize both trade series against the same baseline, then call `make_figure` with normalize="None".
    prices_v = _add_wall_mid(prices)
    bot_v = bot_trades.copy()
    market_v = market_trades.copy()

    if normalize != "None":
        baseline_col = "mid_price" if normalize == "Mid" else "wall_mid"
        prices_v, bot_v = _normalize_by_baseline(prices_v, bot_v, baseline_col=baseline_col)
        prices_v, market_v = _normalize_by_baseline(prices_v, market_v, baseline_col=baseline_col)

    fig = make_figure(
        prices_v,
        bot_v,
        normalize="None",
        levels=levels,
        auto_y_zoom=auto_y_zoom,
        flip_y=flip_y,
        trade_marker_symbol="star",
        trade_trace_name="Bot trades",
    )

    if not market_v.empty:
        trader_v = pd.DataFrame()
        if highlight_trader is not None and highlight_trader != "" and {"buyer", "seller"}.issubset(market_v.columns):
            trader_v = market_v[(market_v["buyer"] == highlight_trader) | (market_v["seller"] == highlight_trader)]
            market_v = market_v.drop(trader_v.index)

        qty = pd.to_numeric(market_v["quantity"], errors="coerce").fillna(0).abs().to_numpy()
        fig.add_trace(
            go.Scattergl(
                x=market_v["timestamp"],
                y=market_v["price"],
                mode="markers",
                name="Market trades",
                marker=dict(
                    color="rgba(0, 0, 0, 0.45)",
                    size=np.clip(10 + 2.6 * np.sqrt(np.maximum(qty, 0)), 10, 28),
                    line=dict(color="rgba(0,0,0,0.55)", width=1),
                    symbol="triangle-up",
                ),
                hovertemplate="t=%{x}<br>px=%{y}<br>qty=%{customdata}<extra></extra>",
                customdata=market_v["quantity"],
            )
        )

        if highlight_trader is not None and not trader_v.empty:
            qty2 = pd.to_numeric(trader_v["quantity"], errors="coerce").fillna(0).abs().to_numpy()
            fig.add_trace(
                go.Scattergl(
                    x=trader_v["timestamp"],
                    y=trader_v["price"],
                    mode="markers",
                    name=f"{highlight_trader} trades",
                    marker=dict(
                        color="rgba(255, 193, 7, 0.95)",
                        size=np.clip(12 + 3.0 * np.sqrt(np.maximum(qty2, 0)), 12, 34),
                        line=dict(color="rgba(0,0,0,0.65)", width=1),
                        symbol="triangle-up",
                    ),
                    hovertemplate="t=%{x}<br>px=%{y}<br>qty=%{customdata}<extra></extra>",
                    customdata=trader_v["quantity"],
                )
            )

    fig.update_layout(title=f"{product}: order book + bot trades")
    st_chart(fig)


def main() -> None:
    st.set_page_config(page_title="IMC Dashboard", layout="wide")

    st.title("IMC Prosperity Dashboard")

    screen = st.sidebar.radio("Screen", options=["Data viewer", "Backtest"], index=0, key="screen")

    if screen == "Data viewer":
        st.subheader("Data viewer")

        data_path = Path(st.sidebar.text_input("Data path (root or round folder)", str(DEFAULT_DATA_ROOT))).expanduser()

        # If user pasted a specific round folder path, use it directly.
        # Otherwise, treat as root and let user select round.
        try:
            data_dir, round_num = _parse_data_path(data_path)
            st.sidebar.caption(f"Detected round folder: `round{round_num}`")
        except Exception:
            rounds = _available_rounds(data_path)
            if not rounds:
                st.error(f"No rounds found under: {data_path} (expected folders like `round2/`)")
                st.stop()
            round_num = st.sidebar.selectbox("Round", options=rounds, index=0, key="data_round")
            data_dir = data_path / f"round{round_num}"

        day_files = _find_round_days(data_dir, int(round_num))
        if not day_files:
            st.error(f"No price/trade files found in: {data_dir}")
            st.stop()

        day_options = [d.day for d in day_files]
        selected_days = st.sidebar.multiselect("Days", options=day_options, default=day_options, key="data_days")
        if not selected_days:
            st.warning("Select at least one day.")
            st.stop()

        # Load and stitch days into one continuous timeline to avoid overlapping timestamps.
        min_day = int(min(selected_days))
        prices_parts: list[pd.DataFrame] = []
        trades_parts: list[pd.DataFrame] = []
        for d in day_files:
            if d.day not in selected_days:
                continue
            p = load_prices(str(d.prices_path)).copy()
            t = load_trades(str(d.trades_path)).copy()

            offset = int(d.day - min_day) * 1_000_000
            p["timestamp_in_day"] = p["timestamp"]
            p["timestamp"] = p["timestamp"] + offset
            t["timestamp_in_day"] = t["timestamp"]
            t["timestamp"] = t["timestamp"] + offset
            t["day"] = d.day

            prices_parts.append(p)
            trades_parts.append(t)

        prices_all = pd.concat(prices_parts, ignore_index=True).sort_values(["product", "timestamp"])
        trades_all = pd.concat(trades_parts, ignore_index=True).sort_values(["symbol", "timestamp"])

        products = sorted(prices_all["product"].unique().tolist())
        product = st.sidebar.selectbox("Product", options=products, index=0, key="data_product")

        # Normalization baseline: None / Mid / Wall Mid
        normalize = st.sidebar.selectbox("Normalize by", options=["None", "Mid", "Wall Mid"], index=0, key="data_norm")
        levels_sel = st.sidebar.multiselect("Book levels", options=[1, 2, 3], default=[1], key="data_book_levels")
        levels = tuple(sorted(int(x) for x in levels_sel)) if levels_sel else (1,)
        auto_y_zoom = st.sidebar.checkbox("Auto y-zoom to book", value=True, help="Zoom y-axis to the bid/ask band in the current window.")
        flip_y = st.sidebar.checkbox("Flip y-axis (bid top)", value=False)

        prices = prices_all[prices_all["product"] == product].sort_values("timestamp").reset_index(drop=True)
        trades = trades_all[trades_all["symbol"] == product].sort_values("timestamp").reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows (prices)", f"{len(prices):,}")
        with c2:
            st.metric("Rows (trades)", f"{len(trades):,}")
        with c3:
            ts_min = int(prices["timestamp"].min()) if len(prices) else None
            ts_max = int(prices["timestamp"].max()) if len(prices) else None
            st.metric("Timestamp range", f"{ts_min} … {ts_max}")

        if len(prices):
            t0, t1 = int(prices["timestamp"].min()), int(prices["timestamp"].max())
            window = st.sidebar.selectbox(
                "Default zoom window", options=[5_000, 10_000, 20_000, 50_000, "All"], index=1, key="data_zoom_window"
            )
            if window == "All":
                default = (t0, t1)
            else:
                w = int(window)
                default = (max(t0, t1 - w), t1)

            sel = st.sidebar.slider("Timestamp window", min_value=t0, max_value=t1, value=default, step=100, key="data_ts_window")
            prices = prices[(prices["timestamp"] >= sel[0]) & (prices["timestamp"] <= sel[1])]
            trades = trades[(trades["timestamp"] >= sel[0]) & (trades["timestamp"] <= sel[1])]

        fig = make_figure(prices, trades, normalize=normalize, levels=levels, auto_y_zoom=auto_y_zoom, flip_y=flip_y)
        st_chart(fig)

        with st.expander("Raw data preview"):
            st.write("Prices (head)")
            st_table(prices.head(50))
            st.write("Trades (head)")
            st_table(trades.head(50))

    elif screen == "Backtest":
        st.subheader("Backtest")
        st.caption(
            "Same engine as `prosperity3bt` (install: `pip install -e backtester`). "
            "Default data directory matches the CLI (`data/` next to `backtester/`, or `./data`)."
        )

        for _k in ("bt_activities", "bt_trades", "bt_market_trades", "bt_sandbox", "bt_error", "bt_meta"):
            if _k not in st.session_state:
                st.session_state[_k] = None

        pnl_full_y = False
        with st.sidebar:
            st.markdown("##### Backtest setup")
            resources_dir = Path(
                st.text_input("Data directory", str(_dashboard_default_data_dir()), help="Same layout as CLI: `roundN/*.csv`", key="bt_res_dir")
            ).expanduser()
            rounds = _available_round_days(resources_dir)
            if not rounds:
                st.warning(f"No `round*/prices_*.csv` under: `{resources_dir}`")
                algo_choice = None
                round_num = None
                day_sel = []
                merge_pnl = True
                match_trades = "all"
            else:
                algo_paths = _algorithms_in_repo(ROOT)
                _names = [p.name for p in algo_paths]
                _default_i = _names.index("Evelyn_v2.py") if "Evelyn_v2.py" in _names else 0
                algo_choice = st.selectbox("Algorithm file", options=algo_paths, format_func=lambda p: p.name, index=_default_i, key="bt_algo")

                round_num = st.selectbox("Round", options=list(rounds.keys()), index=0, key="bt_round")
                days = rounds[round_num]
                day_sel = st.multiselect("Days", options=days, default=days, key="bt_days")

                merge_pnl = st.checkbox(
                    "Merge PnL across days",
                    value=True,
                    help="Adds previous day's ending PnL to the next day.",
                    key="bt_merge_pnl",
                )
                match_trades = st.selectbox("Match trades mode", options=["all", "worse", "none"], index=0, key="bt_match")
                pnl_full_y = st.checkbox(
                    "Full PnL y-axis (include spike outliers)",
                    value=False,
                    key="bt_pnl_full_y",
                    help="Off by default: y-axis uses ~0.5%–99.5% of PnL so typical variation is visible. "
                    "Rare mark-to-market spikes (e.g. 700k+) otherwise flatten the rest of the curve. "
                    "Turn on to plot the full range, or double-click a chart for Plotly autorange.",
                )

            run = st.button("Run backtest", type="primary", use_container_width=True, key="bt_run")
            if st.button("Clear results", use_container_width=True, key="bt_clear"):
                st.session_state.bt_activities = None
                st.session_state.bt_trades = None
                st.session_state.bt_market_trades = None
                st.session_state.bt_sandbox = None
                st.session_state.bt_error = None
                st.session_state.bt_meta = None
                st.rerun()
            if st.button(
                "Reset engine + cached runs",
                use_container_width=True,
                key="bt_reset_engine",
                help="Clears Python’s copy of prosperity3bt and Streamlit’s backtest cache. Use if you still see KeyError on a symbol.",
            ):
                _purge_and_use_repo_prosperity3bt()
                clear_bt = getattr(_run_backtest_cached, "clear", None)
                if callable(clear_bt):
                    clear_bt()
                else:
                    st.cache_data.clear()
                st.session_state.bt_activities = None
                st.session_state.bt_trades = None
                st.session_state.bt_market_trades = None
                st.session_state.bt_sandbox = None
                st.session_state.bt_error = None
                st.session_state.bt_meta = None
                st.rerun()

        if not rounds:
            st.error("Fix the data directory in the sidebar, or create `data/roundN/` CSVs.")
            st.stop()

        if not day_sel:
            st.info("Select at least one day in the sidebar.")
            st.stop()

        if run:
            st.session_state.bt_error = None
            with st.spinner("Running backtest…"):
                try:
                    round_days = tuple((int(round_num), int(d)) for d in sorted(day_sel))
                    activities, trades, sandbox, market_trades = _run_backtest_cached(
                        str(algo_choice),
                        str(resources_dir),
                        round_days,
                        match_trades,
                        merge_pnl,
                    )
                    st.session_state.bt_activities = activities
                    st.session_state.bt_trades = trades
                    st.session_state.bt_market_trades = market_trades
                    st.session_state.bt_sandbox = sandbox
                    st.session_state.bt_meta = {
                        "algorithm": algo_choice.name,
                        "resources": str(resources_dir),
                        "round_days": round_days,
                        "match_trades": match_trades,
                        "merge_pnl": merge_pnl,
                    }
                    st.success("Backtest finished.")
                except Exception as e:
                    st.session_state.bt_activities = None
                    st.session_state.bt_trades = None
                    st.session_state.bt_market_trades = None
                    st.session_state.bt_sandbox = None
                    st.session_state.bt_meta = None
                    st.session_state.bt_error = str(e)

        if st.session_state.bt_error:
            st.error(st.session_state.bt_error)
            err = st.session_state.bt_error
            if "KeyError" in err and ("ASH_COATED" in err or "OSMIUM" in err or "PEPPER" in err):
                st.caption(
                    "This usually means the installed `prosperity3bt` is stale. From the repo root run: "
                    "`pip install -e backtester` (same interpreter Streamlit uses)."
                )

        activities = st.session_state.bt_activities
        trades = st.session_state.bt_trades
        market_trades = st.session_state.bt_market_trades
        sandbox = st.session_state.bt_sandbox
        if activities is not None and trades is not None:
            try:
                import prosperity3bt as _p3

                _origin = getattr(_p3, "__file__", "") or ""
                if _origin and str(ROOT.resolve()) not in _origin.replace("\\", "/"):
                    st.warning(
                        f"The dashboard loaded `prosperity3bt` from `{_origin}`, not this repo’s `backtester/`. "
                        f"**Stop Streamlit completely** (terminal Ctrl+C) and run again with `./.venv/bin/streamlit run dashboard/app.py`, "
                        f"or use **Settings → Clear cached data**. That fixes stale `KeyError` on product names."
                    )
            except Exception:
                pass

            meta = st.session_state.get("bt_meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            if meta:
                st.info(
                    f"**{meta.get('algorithm', '?')}** · days `{meta.get('round_days', ())}` · "
                    f"match **{meta.get('match_trades', '')}** · merge PnL: **{meta.get('merge_pnl', False)}** · "
                    f"`{meta.get('resources', '')}`"
                )

            summary_df, total_pnl = _summary_with_trades(activities, trades)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Products", f"{len(summary_df)}")
            with m2:
                st.metric("Total PnL (final)", f"{total_pnl:,.0f}")
            with m3:
                st.metric("Submission fills (rows)", f"{len(trades):,}")
            with m4:
                n_nonempty = int((sandbox["lambda_log"].astype(str).str.len() > 0).sum()) if sandbox is not None and not sandbox.empty else 0
                st.metric("Timesteps w/ trader output", f"{n_nonempty:,}")

            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button(
                    "Download activities CSV",
                    activities.to_csv(index=False).encode("utf-8"),
                    file_name="backtest_activities.csv",
                    mime="text/csv",
                    key="dl_act",
                )
            with dl2:
                st.download_button(
                    "Download trades CSV",
                    trades.to_csv(index=False).encode("utf-8"),
                    file_name="backtest_trades.csv",
                    mime="text/csv",
                    key="dl_tr",
                )
            with dl3:
                sb_csv = sandbox.to_csv(index=False).encode("utf-8") if sandbox is not None and not sandbox.empty else b"timestamp,lambda_log\n"
                st.download_button(
                    "Download sandbox log CSV",
                    sb_csv,
                    file_name="backtest_sandbox.csv",
                    mime="text/csv",
                    key="dl_sb",
                )

            if market_trades is not None and not market_trades.empty:
                st.download_button(
                    "Download market trades CSV",
                    market_trades.to_csv(index=False).encode("utf-8"),
                    file_name="market_trades.csv",
                    mime="text/csv",
                    key="dl_mkt_tr",
                )

            with st.expander("Final PnL & trade stats by product", expanded=True):
                st_table(summary_df)

            st.caption(
                "**PnL charts:** By default the y-axis is zoomed to the middle of the distribution (~0.5%–99.5%) "
                "so you can see normal variation. A single extreme mark-to-market spike (common on illiquid books) "
                "would otherwise squash everything onto one line. Use the sidebar **Full PnL y-axis** to include spikes, "
                "or double-click a chart for Plotly’s full autorange."
            )
            st_chart(_plot_all_products_pnl(activities, pnl_full_y=pnl_full_y))

            products_bt = sorted(activities["product"].unique().tolist())
            prod = st.selectbox("Product (detail charts)", options=products_bt, index=0, key="bt_product")
            bt_highlight_olivia = st.checkbox(
                "Highlight Olivia trades (yellow triangles)",
                value=True,
                key="bt_highlight_olivia",
                help="Uses buyer/seller columns from `trades_round_*_day_*.csv` to highlight Olivia prints.",
            )
            bt_detail_norm = st.selectbox(
                "Detail chart normalize by",
                options=["None", "Mid", "Wall Mid"],
                index=0,
                key="bt_detail_norm",
            )
            bt_detail_levels_sel = st.multiselect(
                "Detail chart book levels",
                options=[1, 2, 3],
                default=[1],
                key="bt_detail_book_levels",
            )
            bt_detail_levels = tuple(sorted(int(x) for x in bt_detail_levels_sel)) if bt_detail_levels_sel else (1,)
            c_det1, c_det2 = st.columns(2)
            with c_det1:
                bt_detail_auto_y = st.checkbox(
                    "Detail chart auto y-zoom to book",
                    value=True,
                    key="bt_detail_auto_y",
                )
            with c_det2:
                bt_detail_flip_y = st.checkbox(
                    "Detail chart flip y-axis (bid top)",
                    value=False,
                    key="bt_detail_flip_y",
                )

            prod_prices = activities[activities["product"] == prod].sort_values("timestamp").reset_index(drop=True)
            if not prod_prices.empty:
                bt_t0 = int(prod_prices["timestamp"].min())
                bt_t1 = int(prod_prices["timestamp"].max())
                bt_window_choice = st.selectbox(
                    "Detail chart default zoom window",
                    options=[5_000, 10_000, 20_000, 50_000, "All"],
                    index=1,
                    key="bt_detail_zoom_window",
                )
                if bt_window_choice == "All":
                    bt_default = (bt_t0, bt_t1)
                else:
                    bt_w = int(bt_window_choice)
                    bt_default = (max(bt_t0, bt_t1 - bt_w), bt_t1)
                bt_sel = st.slider(
                    "Detail chart timestamp window",
                    min_value=bt_t0,
                    max_value=bt_t1,
                    value=bt_default,
                    step=100,
                    key="bt_detail_ts_window",
                )
                _plot_backtest_book_and_trades(
                    activities,
                    trades,
                    market_trades if market_trades is not None else pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity"]),
                    prod,
                    normalize=bt_detail_norm,
                    levels=bt_detail_levels,
                    auto_y_zoom=bt_detail_auto_y,
                    flip_y=bt_detail_flip_y,
                    window=bt_sel,
                    highlight_trader="Olivia" if bt_highlight_olivia else None,
                )
            _plot_pnl_and_position(activities, trades, prod, pnl_full_y=pnl_full_y)

            if sandbox is not None and not sandbox.empty:
                noisy = sandbox[sandbox["lambda_log"].astype(str).str.len() > 0]
                with st.expander("Trader stdout / print capture (lambda_log)", expanded=False):
                    st.caption(f"Showing {len(noisy):,} non-empty rows (of {len(sandbox):,} timesteps).")
                    st_table(noisy.head(500))

            with st.expander("Raw tables (head)"):
                st.write("Activities")
                st_table(activities.head(200))
                st.write("Trades")
                st_table(trades.head(200))


if __name__ == "__main__":
    main()

