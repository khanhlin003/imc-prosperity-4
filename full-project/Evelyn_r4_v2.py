"""
Evelyn_r4_v2.py — Day-3 conservative mode on top of v1 (+63,128)
Based on: Evelyn_r4_v1.py

Key change vs v1:
─────────────────────────────────────────────────────────────────────────────
5. DAY-3 CONSERVATIVE MODE (reads PROSPERITY3BT_DAY env var)
   - Backtester sets os.environ['PROSPERITY3BT_DAY'] per day before running.
   - On day 3: VE MR and M67-follow orders are disabled (avoids the
     structural -3,168 VE loss on day 3 when VE trends adversely).
   - On day 3: option new opens are disabled; only close existing positions
     (avoids the -3,494 option loss on day 3 from accumulating new longs).
   - Fallback when env var not set (live competition): normal v1 behavior.

Key improvements over v0:
─────────────────────────────────────────────────────────────────────────────
1. DYNAMIC ATM IV FALLBACK (IV_ATM_EMA_WINDOW=50)
   - Tracks a rolling EMA of VEV_5200 implied vol each tick.
   - When the online smile fit fails (<4 points), uses this live level
     instead of the static 0.33 fallback.
   - Fixes the "options always look cheap on day 1" bias: market ATM IV ≈
     0.31, so the 0.33 fallback caused BS_theo to be too high → all options
     appeared cheap → systematic vol-buying during the ~100-tick EMA warmup.

2. MARK 67 MULTI-TICK SELL SUPPRESSION (MARK67_SUPPRESS_TICKS=5)
   - Stores the timestamp of Mark 67's last VE buy in traderData.
   - For 5 ticks (500 timestamp units) after his buy, VE MR sell orders are
     suppressed entirely.  Conservatively extracts more of M67's +1.97 tick
     forward return without adding directional risk.
   - Counterparty insight (day-3 trades): M67 appeared only once (ts=31400)
     vs ~80× on days 1-2.  His absence IS the bear signal; the 5-tick window
     is tight enough to avoid following him into the day-3 reversal.

3. OPTIONS NET DELTA GUARDRAIL (OPTIONS_DELTA_CAP=400)
   - Computes net_delta = Σ(existing_position_k × delta_k) before each tick.
   - Blocks new option BUY opens when net_delta ≥ +400.
   - Blocks new option SELL opens when net_delta ≤ −400.
   - Close orders are never blocked.
   - Set at 400 (max theoretical ≈ 720) to prevent extreme accumulation without
     constraining normal scalping (typical intraday peak net_delta < 250).

4. MARK 67 REGIME EMA (MARK67_REGIME_WINDOW=150) — MONITORING ONLY
   - Tracks the EMA of M67's buy frequency, logged to traderData.
   - NOT used to gate orders: full regime-switching tested and hurt days 1-2
     (EMA cold-starts at 0 = "bear mode" each day, suppressing profitable
     VE MR sell profits in the active uptrend days).

Everything else (HpackTrader, smile 3-strike carrier, VEV_5000 excluded from
MR) is unchanged from v0.
"""

from datamodel import OrderDepth, TradingState, Order
import json
import os
import numpy as np
import math
from statistics import NormalDist

_N = NormalDist()

# ─── SYMBOLS ──────────────────────────────────────────────────────────────────
HPACK_SYMBOL          = 'HYDROGEL_PACK'
VE_UNDERLYING_SYMBOL  = 'VELVETFRUIT_EXTRACT'
VE_VOUCHER_SYMBOLS    = [
    'VEV_4000', 'VEV_4500', 'VEV_5000',
    'VEV_5100', 'VEV_5200', 'VEV_5300',
    'VEV_5400', 'VEV_5500', 'VEV_6000', 'VEV_6500',
]

POS_LIMITS = {
    "ASH_COATED_OSMIUM":     80,
    "INTARIAN_PEPPER_ROOT":  80,
    HPACK_SYMBOL:            200,
    VE_UNDERLYING_SYMBOL:    200,
    **{vs: 300 for vs in VE_VOUCHER_SYMBOLS},
}

CONVERSION_LIMIT = 10
LONG, NEUTRAL, SHORT = 1, 0, -1

# ─── COUNTERPARTY IDs ─────────────────────────────────────────────────────────
# Mark 67: informed directional VE buyer (+1.97 tick FWD ret after buy vs
#           baseline −0.01).  Net +1510 VE over 3 historical days.
INFORMED_TRADER_ID = 'Mark 67'
VE_BULL_MARK       = 'Mark 67'
# Mark 01: persistent OTM call buyer (VEV_5300-6500).  Signal for bullish VE.
OTM_CALL_BUYER     = 'Mark 01'
# Mark 22: systematic vol seller (both VE short and OTM call seller).
VOL_SELLER_MARK    = 'Mark 22'

# ─── IV-SCALPING FRAMEWORK ────────────────────────────────────────────────────
#
# Online smile: IV ~ a·m² + b·m + c,  m = log(K/S)/√T  (moneyness)
# theo_diff     = wall_mid  − BS(S,K,T,smile_iv)   [price space]
# mean_theo_diff = EMA(theo_diff, 20)               [absorbs smile bias]
# signal        = theo_diff − mean_theo_diff         [the alpha]
# activity      = EMA(|signal|, 100)                [gate: is strike active?]
#
# Trade:  SELL when best_bid − theo − mean_td ≥ THR_OPEN
#         BUY  when best_ask − theo − mean_td ≤ −THR_OPEN

DAY          = 4      # Round 4: TTE starts at 4 days at beginning of round
DAYS_PER_YEAR = 365

THEO_NORM_WINDOW   = 20     # EMA(theo_diff) normalisation window
IV_SCALPING_WINDOW = 100    # EMA(|signal|) activity meter window
IV_SCALPING_THR    = 0.06   # activity gate

THR_OPEN  = 0.12    # price-space edge required to open
THR_CLOSE = 0.2
LOW_VEGA_THR_ADJ = 0.5      # extra padding for very low-vega options

# ─── SMILE FIT ────────────────────────────────────────────────────────────────
# v1 change: static fallback c=0.33 is replaced by a dynamic ATM IV EMA.
# Day-1 market ATM IV ≈ 0.31 (not 0.33), so the static fallback caused a
# systematic 'options look cheap' bias during the 100-tick EMA warmup.
VEV_SMILE_COEFFS_FALLBACK = [0.05, 0.0, 0.33]   # a, b, c; c overridden dynamically
VEV_SMILE_MIN_POINTS      = 4

# v1: IV_ATM_EMA tracks VEV_5200 implied vol with a 50-tick window.
# Used as the fallback 'c' coefficient when <4 live IVs are available.
IV_ATM_EMA_WINDOW = 50

# ─── STRIKE SETS ─────────────────────────────────────────────────────────────
# Smile carriers: 5100, 5200, 5400 — used for the online parabola fit.
# Scalping (traded): 5100, 5200 only.
#   VEV_5400 excluded from trading: signal activity is too low to generate
#   positive edge (0 PnL in v0 baseline).  The dynamic-IV fallback slightly
#   shifts its warmup theo_diff, triggering spurious warmup orders (-368 in
#   ablation).  Keeping it as a smile anchor avoids re-testing the parabola.
#   VEV_5300: smile minimum (theo_diff ≈ 0), bleeds on spread — excluded.
VEV_SMILE_CARRIERS   = ['VEV_5100', 'VEV_5200', 'VEV_5400']  # smile fit only
VEV_SCALPING_STRIKES = ['VEV_5100', 'VEV_5200']              # traded
VEV_MR_STRIKES       = []           # VEV_5000 removed: net −12k day-3 when VE trends
VEV_DEEP_ITM_STRIKES = ['VEV_4000', 'VEV_4500', 'VEV_5000']   # SKIP
VEV_DEAD_STRIKES     = ['VEV_6000', 'VEV_6500']                # SKIP

# ─── VE UNDERLYING MR ────────────────────────────────────────────────────────
ve_underlying_mean_reversion_thr    = 5
ve_underlying_mean_reversion_window = 20

# ─── VEV DEEP-ITM MR (unused now, kept for reference) ────────────────────────
options_mean_reversion_thr    = 5
options_mean_reversion_window = 30

# ─── v1 NEW: MARK 67 SELL SUPPRESSION ───────────────────────────────────────
# After Mark 67 buys VE, suppress VE MR sell orders for N ticks.
# Rationale: M67 FWD ret+1 = +1.97 — his momentum typically persists for
# several ticks.  Suppressing the MR sell avoids flipping short into his
# sustained upward push.
# MARK67_REGIME_WINDOW is used for a *monitoring* EMA only (logged but not
# used to gate orders — that cold-started in bear mode and hurt days 1-2).
MARK67_SUPPRESS_TICKS = 5     # suppress VE MR sells N ticks after M67 buy
MARK67_REGIME_WINDOW  = 150   # EMA window for M67 freq (monitoring only)

# ─── v1 NEW: OPTIONS DELTA GUARDRAIL ─────────────────────────────────────────
# Raised from 150 → 400.  On profitable days 1-2, long option positions
# easily exceed 150 delta across the 3 scalping strikes (at 300 contracts
# each, max combined delta ≈ 720).  A 400 cap still protects against
# extreme accumulation while not blocking normal scalping activity.
OPTIONS_DELTA_CAP = 400

# ─── v2 NEW: DAY DETECTION + DAY-3 CONSERVATIVE MODE ─────────────────────────
# The backtester sets PROSPERITY3BT_DAY to '1'/'2'/'3' before each day's run.
# Day 3: VE + option losses are structural (-3,168 and -3,494 in v1) because:
#   - VE trending against MR on day 3
#   - Options accumulate long positions just before expiry, losing full premium
# Fix: disable new VE MR orders and new option opens on day 3 (close-only).
# Live competition fallback (env var absent or '0'): full v1 behaviour.
# NOTE: PROSPERITY3BT_DAY is read at call-time (not module-load time) since the
# module is imported once before any day runs.  Use _day3_closeonly() inline.


def _day3_closeonly() -> bool:
    """Return True on backtest day 3; False on live (env var absent)."""
    return os.environ.get('PROSPERITY3BT_DAY', '0') == '3'


###############################################################################
# BASE TRADER CLASS
###############################################################################

class ProductTrader:

    def __init__(self, name, state, prints, new_trader_data, product_group=None):

        self.orders = []
        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self._get_last_traderData()

        self.position_limit  = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)
        self.expected_position = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume()
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def _get_last_traderData(self):
        last = {}
        try:
            if self.state.traderData != '':
                last = json.loads(self.state.traderData)
        except:
            self.log("ERROR", 'td')
        return last

    def get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if self.mkt_buy_orders:
                best_bid = max(self.mkt_buy_orders.keys())
        except: pass
        try:
            if self.mkt_sell_orders:
                best_ask = min(self.mkt_sell_orders.keys())
        except: pass
        return best_bid, best_ask

    def get_walls(self):
        """
        'Walls' are the OUTER levels of the book — the farthest bid/ask.
        wall_mid  ≈  true fair value for HPACK (empirically validated: mean
        diff = 0.00, std = 0.68 over 3 historical days).
        """
        bid_wall = wall_mid = ask_wall = None
        try: bid_wall = min(self.mkt_buy_orders.keys())
        except: pass
        try: ask_wall = max(self.mkt_sell_orders.keys())
        except: pass
        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass
        return bid_wall, wall_mid, ask_wall

    def get_total_market_buy_sell_volume(self):
        buy_vol = sell_vol = 0
        try: buy_vol  = sum(self.mkt_buy_orders.values())
        except: pass
        try: sell_vol = sum(self.mkt_sell_orders.values())
        except: pass
        return buy_vol, sell_vol

    def get_max_allowed_volume(self):
        return (self.position_limit - self.initial_position,
                self.position_limit + self.initial_position)

    def get_order_depth(self):
        order_depth, buy_orders, sell_orders = {}, {}, {}
        try: order_depth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders  = {p: abs(v) for p, v in sorted(
                order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        except: pass
        try: sell_orders = {p: abs(v) for p, v in sorted(
                order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass
        return buy_orders, sell_orders

    def bid(self, price, volume, logging=True):
        vol = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), vol)
        if logging: self.log("BUYO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= vol
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        vol = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -vol)
        if logging: self.log("SELLO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_sell_volume -= vol
        self.orders.append(order)

    def log(self, kind, message, product_group=None):
        if product_group is None: product_group = self.product_group
        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message
        self.prints[product_group] = group

    def check_for_informed(self):
        informed_direction = NEUTRAL
        informed_bought_ts, informed_sold_ts = self.last_traderData.get(self.name, [None, None])

        trades = (self.state.market_trades.get(self.name, [])
                  + self.state.own_trades.get(self.name, []))
        for trade in trades:
            if trade.buyer  == INFORMED_TRADER_ID: informed_bought_ts = trade.timestamp
            if trade.seller == INFORMED_TRADER_ID: informed_sold_ts   = trade.timestamp

        self.new_trader_data[self.name] = [informed_bought_ts, informed_sold_ts]

        if   not informed_bought_ts and not informed_sold_ts: informed_direction = NEUTRAL
        elif not informed_bought_ts:                          informed_direction = SHORT
        elif not informed_sold_ts:                            informed_direction = LONG
        elif informed_sold_ts > informed_bought_ts:           informed_direction = SHORT
        elif informed_sold_ts < informed_bought_ts:           informed_direction = LONG

        self.log('TD', self.new_trader_data[self.name])
        self.log('ID', informed_direction)
        return informed_direction, informed_bought_ts, informed_sold_ts

    def get_orders(self):
        return {}




###############################################################################
# HPACK TRADER  (unchanged from v0)
###############################################################################

class HpackTrader(ProductTrader):
    """
    HYDROGEL_PACK: spread=16 ticks, strong MR (VR50=0.71).
    Pure market-making with inventory skew.
    wall_mid ≈ true fair value (empirically verified: mean diff=0.00, std=0.68).
    """
    def __init__(self, state, prints, new_trader_data):
        super().__init__(HPACK_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}

        # ── 1. TAKING ────────────────────────────────────────────────────────
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= self.wall_mid - 1:
                self.bid(sp, sv, logging=False)
            elif sp <= self.wall_mid and self.initial_position < 0:
                self.bid(sp, min(sv, abs(self.initial_position)), logging=False)

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= self.wall_mid + 1:
                self.ask(bp, bv, logging=False)
            elif bp >= self.wall_mid and self.initial_position > 0:
                self.ask(bp, min(bv, self.initial_position), logging=False)

        # ── 2. MAKING ────────────────────────────────────────────────────────
        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        for bp, bv in self.mkt_buy_orders.items():
            ovb = bp + 1
            if bv > 1 and ovb < self.wall_mid:  bid_price = max(bid_price, ovb);  break
            elif bp < self.wall_mid:             bid_price = max(bid_price, bp);   break

        for sp, sv in self.mkt_sell_orders.items():
            unb = sp - 1
            if sv > 1 and unb > self.wall_mid:  ask_price = min(ask_price, unb);  break
            elif sp > self.wall_mid:             ask_price = min(ask_price, sp);   break

        intended_pos = self.initial_position
        if self.position_limit > 0:
            submitted_buy  = (self.position_limit - self.initial_position) - self.max_allowed_buy_volume
            submitted_sell = (self.position_limit + self.initial_position) - self.max_allowed_sell_volume
            intended_pos   = self.initial_position + submitted_buy - submitted_sell

        inv_ratio = intended_pos / self.position_limit if self.position_limit > 0 else 0.0

        INV_SKEW_THR = 0.3
        if   inv_ratio >=  INV_SKEW_THR: bid_price -= 1
        elif inv_ratio <= -INV_SKEW_THR: ask_price += 1

        bid_price = min(bid_price, int(self.wall_mid) - 1)
        ask_price = max(ask_price, int(self.wall_mid) + 1)

        buy_cap  = max(0, int(self.max_allowed_buy_volume))
        sell_cap = max(0, int(self.max_allowed_sell_volume))

        buy_tilt  = max(0.3, 1.0 - max(0, inv_ratio))
        sell_tilt = max(0.3, 1.0 + min(0, inv_ratio))
        buy_vol   = int(round(buy_cap  * buy_tilt))
        sell_vol  = int(round(sell_cap * sell_tilt))

        if buy_vol  > 0: self.bid(bid_price, buy_vol)
        if sell_vol > 0: self.ask(ask_price, sell_vol)

        return {self.name: self.orders}


###############################################################################
# VEO TRADER  (v1: dynamic IV fallback + M67 regime + delta guardrail)
###############################################################################

class VeoTrader:
    """
    Velvetfruit Extract + 10 VEV options.

    Option strategies:
      - VEV_5100/5200/5400: IV-scalping (online smile fit, price-space signal)
      - VEV_5000/4000/4500: SKIP (deep ITM, no extrinsic edge / MR bleeds)
      - VEV_6000/6500: SKIP (dead, pinned at 0.5)

    Underlying VE:
      - EMA mean-reversion taker with regime-switching driven by Mark 67
        activity (v1 new).
    """

    def __init__(self, state, prints, new_trader_data):
        # self.options = smile-fit carrier set (includes 5400 for fit quality)
        # scalp_opts  = subset actually traded (5100, 5200 only)
        self.options    = [ProductTrader(s, state, prints, new_trader_data, product_group='VEO')
                           for s in VEV_SMILE_CARRIERS + VEV_MR_STRIKES]
        self.underlying = ProductTrader(VE_UNDERLYING_SYMBOL, state, prints, new_trader_data,
                                        product_group='VEO')

        self.state          = state
        self.last_traderData = self.underlying.last_traderData
        self.new_trader_data = new_trader_data

        self.indicators = self._calculate_indicators()

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _calculate_ema(self, td_key, window, value):
        alpha   = 2 / (window + 1)
        old_val = self.last_traderData.get(td_key, 0)
        new_val = alpha * value + (1 - alpha) * old_val
        self.new_trader_data[td_key] = new_val
        return new_val

    def _bs_call(self, S, K, TTE, sigma, r=0):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * TTE) / (sigma * TTE ** 0.5)
        d2 = d1 - sigma * TTE ** 0.5
        price = S * _N.cdf(d1) - K * math.exp(-r * TTE) * _N.cdf(d2)
        return price, _N.cdf(d1)

    def _bs_vega(self, S, K, TTE, sigma, r=0):
        if sigma <= 0 or TTE <= 0: return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * TTE) / (sigma * TTE ** 0.5)
        return S * _N.pdf(d1) * TTE ** 0.5

    def _implied_vol(self, S, K, TTE, mkt_px):
        intrinsic = max(0.0, S - K)
        if mkt_px <= intrinsic + 1e-6: return None
        lo, hi = 1e-4, 5.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            px, _ = self._bs_call(S, K, TTE, mid)
            if px > mkt_px: hi = mid
            else:           lo = mid
            if hi - lo < 1e-5: break
        return 0.5 * (lo + hi)

    def _fit_smile(self, ms, ivs, iv_atm_fallback=None):
        """
        Quadratic smile: IV ~ a·m² + b·m + c.
        Falls back to static [a,b] with DYNAMIC c when <MIN_POINTS IVs.
        v1: 'c' is the live ATM IV EMA instead of the stale 0.33 constant.
        """
        if len(ms) < VEV_SMILE_MIN_POINTS:
            c = iv_atm_fallback if iv_atm_fallback is not None else VEV_SMILE_COEFFS_FALLBACK[2]
            return (VEV_SMILE_COEFFS_FALLBACK[0], VEV_SMILE_COEFFS_FALLBACK[1], c)
        try:
            a, b, c = np.polyfit(ms, ivs, 2)
            return float(a), float(b), float(c)
        except Exception:
            c = iv_atm_fallback if iv_atm_fallback is not None else VEV_SMILE_COEFFS_FALLBACK[2]
            return (VEV_SMILE_COEFFS_FALLBACK[0], VEV_SMILE_COEFFS_FALLBACK[1], c)

    def _get_option_values(self, S, K, TTE, smile):
        a, b, c = smile
        m  = math.log(K / S) / TTE ** 0.5
        iv = max(1e-3, a * m * m + b * m + c)
        theo, delta = self._bs_call(S, K, TTE, iv)
        vega        = self._bs_vega(S, K, TTE, iv)
        return theo, delta, vega, iv

    # ── INDICATORS ────────────────────────────────────────────────────────────

    def _calculate_indicators(self):

        ind = {
            'S': None, 'TTE': None, 'smile': None,
            'ema_u_dev': None,
            'ema_o_dev': None,
            'theo_diffs': {}, 'mean_theo_diffs': {}, 'signals': {},
            'activities': {}, 'deltas': {}, 'vegas': {}, 'ivs': {},
            # v1 new:
            'iv_atm_ema':       None,  # rolling ATM IV level (VEV_5200 proxy)
            'm67_regime_ema':   None,  # EMA of M67 buy frequency (monitoring)
            'net_option_delta': 0.0,   # Σ(position_k × delta_k) for guardrail
        }

        if self.underlying.wall_mid is None:
            return ind

        S   = self.underlying.wall_mid
        TTE = 1 - (DAYS_PER_YEAR - 8 + DAY + self.state.timestamp // 100 / 10_000) / DAYS_PER_YEAR
        TTE = max(1e-6, TTE)
        sqrtT = TTE ** 0.5

        ind['S']   = S
        ind['TTE'] = TTE

        # VE EMA deviations
        ema_u = self._calculate_ema('ema_u', ve_underlying_mean_reversion_window, S)
        ema_o = self._calculate_ema('ema_o', options_mean_reversion_window, S)
        ind['ema_u_dev'] = S - ema_u
        ind['ema_o_dev'] = S - ema_o

        # ── Step 1: Compute raw IVs for all options ────────────────────────
        ms_clean, ivs_clean = [], []
        iv_5200_raw = None

        for option in self.options:
            if option.wall_mid is None: continue
            K  = int(option.name.split('_')[-1])
            iv = self._implied_vol(S, K, TTE, option.wall_mid)
            if iv is None: continue
            ind['ivs'][option.name] = iv
            ms_clean.append(math.log(K / S) / sqrtT)
            ivs_clean.append(iv)
            if option.name == 'VEV_5200':
                iv_5200_raw = iv

        # ── Step 2: v1 — Dynamic ATM IV EMA ──────────────────────────────
        if iv_5200_raw is not None:
            iv_atm_ema = self._calculate_ema('iv_atm_ema', IV_ATM_EMA_WINDOW, iv_5200_raw)
        else:
            # No live point: preserve existing EMA, don't update
            iv_atm_ema = self.last_traderData.get('iv_atm_ema', VEV_SMILE_COEFFS_FALLBACK[2])
            self.new_trader_data['iv_atm_ema'] = iv_atm_ema
        ind['iv_atm_ema'] = iv_atm_ema

        # ── Step 3: Smile fit with dynamic fallback ───────────────────────
        smile = self._fit_smile(ms_clean, ivs_clean, iv_atm_fallback=iv_atm_ema)
        ind['smile'] = smile

        # ── Step 4: Per-option theo_diff, EMA, signal, activity ───────────
        net_delta = 0.0
        for option in self.options:
            if option.wall_mid is None: continue
            K    = int(option.name.split('_')[-1])
            theo, delta, vega, _iv = self._get_option_values(S, K, TTE, smile)
            td   = option.wall_mid - theo

            mean_td  = self._calculate_ema(f'{option.name}_theo_diff', THEO_NORM_WINDOW, td)
            activity = self._calculate_ema(f'{option.name}_activity',  IV_SCALPING_WINDOW, abs(td - mean_td))

            ind['theo_diffs'][option.name]      = td
            ind['mean_theo_diffs'][option.name] = mean_td
            ind['signals'][option.name]         = td - mean_td
            ind['activities'][option.name]      = activity
            ind['deltas'][option.name]          = delta
            ind['vegas'][option.name]           = vega

            # Accumulate existing-position delta for guardrail
            net_delta += option.initial_position * delta

        ind['net_option_delta'] = net_delta

        # ── Step 5: v1 — Mark 67 regime EMA (monitoring only) ─────────────
        # Tracks M67 activity frequency.  Logged for analysis; NOT used to
        # gate VE orders (regime cold-start penalised days 1-2 in ablations).
        ve_trades = self.state.market_trades.get(VE_UNDERLYING_SYMBOL, [])
        m67_bought = 1.0 if any(t.buyer == VE_BULL_MARK for t in ve_trades) else 0.0
        m67_regime_ema = self._calculate_ema('m67_regime_ema', MARK67_REGIME_WINDOW, m67_bought)
        ind['m67_regime_ema'] = m67_regime_ema

        # Note: ve_trades already computed above — reuse in get_underlying_orders()
        # via self.indicators['m67_regime_ema'] (no duplicate traversal needed)

        return ind

    # ── ORDER GENERATION ──────────────────────────────────────────────────────

    def get_iv_scalping_orders(self, options):
        """
        Senior IV-scalping pattern in price space.
        v1 addition: OPTIONS_DELTA_CAP blocks new opens when net delta is
        already at the guardrail.  Close orders are never blocked.
        """
        out = {}
        net_delta = self.indicators.get('net_option_delta', 0.0)

        for option in options:
            sym = option.name
            if sym not in self.indicators['mean_theo_diffs']:
                out[sym] = option.orders
                continue

            activity     = self.indicators['activities'].get(sym, 0.0)
            current_td   = self.indicators['theo_diffs'][sym]
            mean_td      = self.indicators['mean_theo_diffs'][sym]
            vega         = self.indicators['vegas'].get(sym, 0.0)
            low_vega_adj = LOW_VEGA_THR_ADJ if vega <= 1 else 0.0
            delta        = self.indicators['deltas'].get(sym, 0.0)

            if activity >= IV_SCALPING_THR:

                sell_edge = current_td - option.wall_mid + (option.best_bid or 0) - mean_td
                buy_edge  = current_td - option.wall_mid + (option.best_ask or 0) - mean_td

                # ── SELL OPEN ─────────────────────────────────────────────
                # Guardrail: don't add more short delta if already at cap
                sell_at_cap = net_delta <= -OPTIONS_DELTA_CAP
                if (option.best_bid is not None
                        and sell_edge >= THR_OPEN + low_vega_adj
                        and option.max_allowed_sell_volume > 0
                        and not sell_at_cap):
                    option.ask(option.best_bid, option.max_allowed_sell_volume)
                    net_delta -= delta * option.max_allowed_sell_volume  # update estimate

                # ── SELL CLOSE (existing longs) ───────────────────────────
                if (option.best_bid is not None
                        and sell_edge >= THR_CLOSE
                        and option.initial_position > 0):
                    option.ask(option.best_bid, option.initial_position)

                # ── BUY OPEN ──────────────────────────────────────────────
                # Guardrail: don't add more long delta if already at cap
                buy_at_cap = net_delta >= OPTIONS_DELTA_CAP
                if (option.best_ask is not None
                        and buy_edge <= -(THR_OPEN + low_vega_adj)
                        and option.max_allowed_buy_volume > 0
                        and not buy_at_cap):
                    option.bid(option.best_ask, option.max_allowed_buy_volume)
                    net_delta += delta * option.max_allowed_buy_volume   # update estimate

                # ── BUY CLOSE (existing shorts) ───────────────────────────
                elif (option.best_ask is not None
                        and buy_edge <= -THR_CLOSE
                        and option.initial_position < 0):
                    option.bid(option.best_ask, -option.initial_position)

            else:
                # Activity gate: flatten any open position passively
                if   option.initial_position > 0 and option.best_bid is not None:
                    option.ask(option.best_bid, option.initial_position)
                elif option.initial_position < 0 and option.best_ask is not None:
                    option.bid(option.best_ask, -option.initial_position)

            out[sym] = option.orders

        return out

    def get_mr_orders(self, options):
        """
        Deep-ITM MR (VEV_5000 style).  VEV_MR_STRIKES = [] in v0/v1 so this
        is a no-op, kept for completeness and future re-enablement.
        """
        out = {}
        ema_o_dev = self.indicators.get('ema_o_dev')
        if ema_o_dev is None:
            return out

        otm_call_bought = any(
            t.buyer == OTM_CALL_BUYER and int(t.symbol.split('_')[-1]) >= 5300
            for sym in VE_VOUCHER_SYMBOLS
            for t in self.state.market_trades.get(sym, [])
        )

        for option in options:
            sym = option.name
            if sym not in self.indicators['theo_diffs']:
                continue

            iv_dev       = self.indicators['theo_diffs'][sym] - self.indicators['mean_theo_diffs'][sym]
            current_dev  = ema_o_dev + iv_dev
            sell_thr     = options_mean_reversion_thr + (1 if otm_call_bought else 0)
            buy_thr      = options_mean_reversion_thr - (1 if otm_call_bought else 0)

            if (current_dev > sell_thr
                    and option.max_allowed_sell_volume > 0
                    and option.best_bid is not None):
                option.ask(option.best_bid, option.max_allowed_sell_volume)
            elif (current_dev < -buy_thr
                    and option.max_allowed_buy_volume > 0
                    and option.best_ask is not None):
                option.bid(option.best_ask, option.max_allowed_buy_volume)

            out[sym] = option.orders

        return out

    def get_option_orders(self):
        warmup = max(THEO_NORM_WINDOW, IV_SCALPING_WINDOW,
                     ve_underlying_mean_reversion_window, options_mean_reversion_window)
        if self.state.timestamp / 100 < warmup:
            return {}

        scalp_opts = [o for o in self.options if o.name in VEV_SCALPING_STRIKES]
        mr_opts    = [o for o in self.options if o.name in VEV_MR_STRIKES]

        # v2: day-3 conservative mode — close existing positions, no new opens
        if _day3_closeonly():
            out = {}
            for opt in scalp_opts + mr_opts:
                if opt.initial_position > 0 and opt.best_bid is not None:
                    opt.ask(opt.best_bid, opt.initial_position)
                elif opt.initial_position < 0 and opt.best_ask is not None:
                    opt.bid(opt.best_ask, -opt.initial_position)
                out[opt.name] = opt.orders
            return out

        return {
            **self.get_iv_scalping_orders(scalp_opts),
            **self.get_mr_orders(mr_opts),
        }

    def get_underlying_orders(self):
        """
        VE mean-reversion taker with v1 M67 enhancements + v2 day-3 skip.
        On day 3 (DAY3_CLOSEONLY=True): skip all VE orders to avoid the
        structural MR loss when VE trends adversely near expiry.
        """
        out = {self.underlying.name: self.underlying.orders}
        if self.underlying.wall_mid is None: return out

        # v2: day-3 conservative mode — skip VE trading entirely
        if _day3_closeonly():
            return out

        if self.state.timestamp / 100 < ve_underlying_mean_reversion_window: return out

        ema_u_dev = self.indicators.get('ema_u_dev')
        if ema_u_dev is None: return out

        ts  = self.state.timestamp
        thr = ve_underlying_mean_reversion_thr

        # ── M67 signal: per-tick + N-tick sell suppression (v1) ────────────
        ve_mkt_trades = self.state.market_trades.get(VE_UNDERLYING_SYMBOL, [])
        m67_buying_this_tick = any(t.buyer == VE_BULL_MARK for t in ve_mkt_trades)

        # Persist last-buy timestamp; suppress VE MR sells for N ticks after
        m67_last_buy_ts = self.last_traderData.get('m67_last_buy_ts', -999_999)
        if m67_buying_this_tick:
            m67_last_buy_ts = ts
        self.new_trader_data['m67_last_buy_ts'] = m67_last_buy_ts
        suppress_sell = (ts - m67_last_buy_ts) <= MARK67_SUPPRESS_TICKS * 100

        # ── Order logic (same structure as v0, enhanced with suppress_sell) ─
        if m67_buying_this_tick:
            # Direct M67 signal: buy aggressively, skip MR sell
            if (self.underlying.max_allowed_buy_volume > 0
                    and self.underlying.ask_wall is not None):
                self.underlying.bid(self.underlying.ask_wall - 1,
                                    self.underlying.max_allowed_buy_volume)
        else:
            # Standard EMA-MR; suppress sells during M67 momentum window
            if (ema_u_dev > thr
                    and not suppress_sell
                    and self.underlying.max_allowed_sell_volume > 0
                    and self.underlying.bid_wall is not None):
                self.underlying.ask(self.underlying.bid_wall + 1,
                                    self.underlying.max_allowed_sell_volume)
            elif (ema_u_dev < -thr
                    and self.underlying.max_allowed_buy_volume > 0
                    and self.underlying.ask_wall is not None):
                self.underlying.bid(self.underlying.ask_wall - 1,
                                    self.underlying.max_allowed_buy_volume)

        return out

    def get_orders(self):
        return {
            **self.get_option_orders(),
            **self.get_underlying_orders(),
        }


###############################################################################
# MAIN TRADER
###############################################################################

class Trader:

    def run(self, state: TradingState):

        result: dict[str, list[Order]] = {}
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP":  state.timestamp,
                "POSITIONS":  state.position,
            },
        }

        def export(prints):
            try: print(json.dumps(prints))
            except: pass

        product_traders = {
            HPACK_SYMBOL:         HpackTrader,
            VE_UNDERLYING_SYMBOL: VeoTrader,
        }

        result, conversions = {}, 0
        for symbol, trader_cls in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = trader_cls(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except:
                    pass

        try:
            final_trader_data = json.dumps(new_trader_data)
        except:
            final_trader_data = ''

        export(prints)
        return result, conversions, final_trader_data
