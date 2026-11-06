from datamodel import OrderDepth, TradingState, Order
import json
import numpy as np
import math
from statistics import NormalDist

_N = NormalDist()

####### GENERAL ####### GENERAL ####### GENERAL ####### GENERAL ####### GENERAL ####### GENERAL ####### GENERAL ####### GENERAL  

# Round 3
HPACK_SYMBOL = 'HYDROGEL_PACK'

VE_UNDERLYING_SYMBOL = 'VELVETFRUIT_EXTRACT'

VE_VOUCHER_SYMBOLS = [
    'VEV_4000',
    'VEV_4500',
    'VEV_5000',
    'VEV_5100',
    'VEV_5200',
    'VEV_5300',
    'VEV_5400',
    'VEV_5500',
    'VEV_6000',
    'VEV_6500',
]

POS_LIMITS = {

    # Local products
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,

    # Round 3
    HPACK_SYMBOL: 200,
    VE_UNDERLYING_SYMBOL: 200,
    **{vs: 300 for vs in VE_VOUCHER_SYMBOLS},
}

CONVERSION_LIMIT = 10

LONG, NEUTRAL, SHORT = 1, 0, -1


INFORMED_TRADER_ID = 'Olivia'



####### VEV / VOUCHERS — DATA CHARACTERISTICS #######
#
# Numbers below come from 3 historical days of round-3 prices, verified with
# analyze_vev.py and analyze_ve_underlying.py.  Strategy is modelled on the
# Prosperity-3 IV-SCALPING approach (Frankfurt Hedgehogs):
#
#   theo            = BS(S, K, T, iv_smile)            # smile = online quad fit
#   theo_diff       = wall_mid - theo                  # PRICE-space deviation
#   mean_theo_diff  = EMA(theo_diff, 20)               # absorbs the smile bias
#   signal          = theo_diff - mean_theo_diff       # the alpha
#   activity        = EMA(|signal|, 100)               # is the strike worth trading?
#
# We trade the SHORT-TERM mean reversion of `signal`, gated by `activity`.
# Lag-1 autocorr of theo_diff returns is -0.41 to -0.50 across active strikes,
# which is essentially the same regime the senior team scalped in 2025.
#
# VELVETFRUIT_EXTRACT (delta-1 underlying)
#   - tight 5-tick spread, ~38 each side
#   - lag-1 autocorr of returns ≈ -0.04  (WEAK MR — much weaker than Squid Ink)
#   - EMA(20) dev: max ≈ 7.5, q95 ≈ 4.   Threshold 5 fires ~1% of the time.
#
# VEV_5000..VEV_5500 — ACTIVE
#   - IV mean ≈ 0.33 across days (drifts 0.32 -> 0.36) — refit smile online
#   - signal_std ≈ 0.22-0.31, signal_max_abs ≈ 0.85-1.30
#   - activity (EMA |signal|, win=100) typical level ≈ 0.13-0.25 — much
#     smaller than the senior team's 0.7 threshold; we use 0.05 (effectively
#     always on; the gate exists mainly to flatten on a true dead patch).
#   - Vegas: 5000=28, 5100=95, 5200=184, 5300=187, 5400=102, 5500=35.
#
# VEV_4000 / VEV_4500 — DEEP ITM
#   - Pinned at intrinsic ± 10.  No extrinsic edge.  SKIP.
#
# VEV_6000 / VEV_6500 — DEAD
#   - Pinned at 0.5 every tick.  SKIP.
#
####### VEV OPTIONS — KNOBS ####### VEV OPTIONS — KNOBS ####### VEV OPTIONS — KNOBS  

DAY = 3              # round 3 live = day 3 of a 7-day expiry  ->  TTE_start = 5d
DAYS_PER_YEAR = 365

# IV-SCALPING (senior's framework, recalibrated for VEV).
#
# After tuning across all 3 historical days:
#   IV_SCALPING_THR=0.05  (the gate barely bites; our activities ~0.13-0.25)
#   THR_OPEN=0.1          (signal_std≈0.25; 0.4σ open => trade ~half the ticks)
#   THR_CLOSE=0.2         (close earlier than open => take quick exits)
THEO_NORM_WINDOW   = 20    # EMA window for mean(theo_diff) normalization
IV_SCALPING_WINDOW = 100   # EMA window for the activity meter
IV_SCALPING_THR    = 0.05

# Open / close edge thresholds in PRICE units:
#   edge_sell = best_bid - theo - mean_theo_diff
#   edge_buy  = best_ask - theo - mean_theo_diff
# Open a sell at edge_sell >= THR_OPEN, scale out at THR_CLOSE.
THR_OPEN  = 0.1
THR_CLOSE = 0.0   # close at any slight reversal (edge >= 0)
LOW_VEGA_THR_ADJ = 0.5     # extra padding when vega <= 1 (rare for VEV)

# Online smile fit.
# The smile CARRIER SET (data points for the fit) is deliberately wider than
# the TRADING SET.  5300 and 5500 bleed when scalped (ablation: -3.5k) but
# their IV observations are valid and add crucial OTM coverage to the fit.
# Without them, the parabola has only 1 degree of freedom (4 pts, 3 params)
# and is asymmetric (3 ITM pts, 1 barely-OTM) — one noisy tick can swing
# all theos simultaneously.
#
# Fit uses 6 strikes (4 ITM/ATM + 2 OTM) → 3 DoF, symmetric coverage.
# Fitted [a,b,c] are then EMA-smoothed (window=10) to dampen tick noise.
VEV_SMILE_COEFFS_FALLBACK = [0.05, 0.0, 0.33]
VEV_SMILE_MIN_POINTS      = 4   # minimum clean IVs before falling back
VEV_SMILE_EMA_WINDOW      = 10  # EMA window to smooth a, b, c across ticks
VEV_SMILE_MIN_M_SPREAD    = 0.3 # reject fit if moneyness range is too narrow

# Order generation:
#   Scalping: 5100, 5200, 5400 -- the strikes that trade profitably.  5300
#     consistently bleeds (sits at the smile minimum) and 5500 has a
#     structural rich bias the parabola cannot absorb -- both excluded.
#   MR     : 5000 only -- senior's 9500 analog (deep-ITM-ish, behaves
#     delta-1, used as a price-space MR target combined with VE ema_o_dev).
#   Smile carriers (passive, not traded): 5300 + 5500 included for wider
#     moneyness coverage in the quadratic fit only.
VEV_SCALPING_STRIKES = ['VEV_5100', 'VEV_5200', 'VEV_5400']
VEV_MR_STRIKES       = ['VEV_5000']
VEV_SMILE_ONLY       = ['VEV_5300', 'VEV_5500']  # passive carriers — widen smile fit, never traded
# `self.options` (smile-fit carrier set) is built from these two lists.
VEV_DEEP_ITM_STRIKES = ['VEV_4000', 'VEV_4500']    # SKIP — pinned at intrinsic
VEV_DEAD_STRIKES     = ['VEV_6000', 'VEV_6500']    # SKIP — pinned at 0.5

# Underlying VE: lightweight MR taker.  ac1(returns) ≈ -0.04 so alpha is
# small.  EMA(20) dev maxes at ~7.5, so threshold 5 is a reasonable upper
# tail (fires ~1.2% of the time).
ve_underlying_mean_reversion_thr    = 5
ve_underlying_mean_reversion_window = 20

# Deep-ITM-option MR (senior style: ema_o_dev + iv_deviation).
options_mean_reversion_thr    = 5
options_mean_reversion_window = 30

# (No explicit delta hedging — IV scalping holds positions for short bursts;
# senior team explicitly avoided it because of bid-ask spread cost.)


####### HYDROGEL_PACK ####### HYDROGEL_PACK ####### HYDROGEL_PACK ####### HYDROGEL_PACK ####### HYDROGEL_PACK  

# HYDROGEL_PACK (delta-1, standalone)
#   - Mean-reverting:  lag-1 autocorr -0.13;  VR(50) = 0.71  (<<1).
#   - Spread fixed at 16 about 92% of the time -> classic MM target.
#   - Uncorrelated with VELVETFRUIT_EXTRACT (corr ≈ 0.006);
#     trade it standalone, no cross-hedging needed.
#   - Strategy: identical to RAINFOREST_RESIN (StaticTrader pattern):
#     take 1-tick edges off the wall_mid, then quote inside the walls.

class ProductTrader:

    def __init__(self, name, state, prints, new_trader_data, product_group=None):

        self.orders = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self.get_last_traderData()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0) # position at beginning of round

        self.expected_position = self.initial_position # update this if you expect a certain change in position e.g. to already hedge


        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume() # gets updated when order created
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def get_last_traderData(self):
                        
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except: self.log("ERROR", 'td')

        return last_traderData


    def get_best_bid_ask(self):

        best_bid = best_ask = None

        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except: pass

        return best_bid, best_ask


    def get_walls(self):

        bid_wall = wall_mid = ask_wall = None

        try: bid_wall = min([x for x,_ in self.mkt_buy_orders.items()])
        except: pass
        
        try: ask_wall = max([x for x,_ in self.mkt_sell_orders.items()])
        except: pass

        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass

        return bid_wall, wall_mid, ask_wall
    
    def get_total_market_buy_sell_volume(self):

        market_bid_volume = market_ask_volume = 0

        try:
            market_bid_volume = sum([v for p, v in self.mkt_buy_orders.items()])
            market_ask_volume = sum([v for p, v in self.mkt_sell_orders.items()])
        except: pass

        return market_bid_volume, market_ask_volume
    

    def get_max_allowed_volume(self):
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def get_order_depth(self):

        order_depth, buy_orders, sell_orders = {}, {}, {}

        try: order_depth: OrderDepth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        except: pass
        try: sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass

        return buy_orders, sell_orders
    

    def bid(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging: self.log("BUYO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        if logging: self.log("SELLO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
        self.max_allowed_sell_volume -= abs_volume
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

        informed_direction, informed_bought_ts, informed_sold_ts = NEUTRAL, None, None
        

        informed_bought_ts, informed_sold_ts = self.last_traderData.get(self.name, [None, None])

        trades = self.state.market_trades.get(self.name, []) + self.state.own_trades.get(self.name, [])

        for trade in trades:
            if trade.buyer == INFORMED_TRADER_ID:
                informed_bought_ts = trade.timestamp
            if trade.seller == INFORMED_TRADER_ID: 
                informed_sold_ts = trade.timestamp
            
        self.new_trader_data[self.name] = [informed_bought_ts, informed_sold_ts]

        informed_sold = informed_sold_ts is not None
        informed_bought = informed_bought_ts is not None

        if not informed_bought and not informed_sold:
            informed_direction = NEUTRAL

        elif not informed_bought and informed_sold:
            informed_direction = SHORT

        elif informed_bought and not informed_sold:
            informed_direction = LONG

        elif informed_bought and informed_sold:
            if informed_sold_ts > informed_bought_ts:
                informed_direction = SHORT
            elif informed_sold_ts < informed_bought_ts:
                informed_direction = LONG
            else:
                informed_direction = NEUTRAL

        self.log('TD', self.new_trader_data[self.name])
        self.log('ID', informed_direction)

        return informed_direction, informed_bought_ts, informed_sold_ts


    def get_orders(self):
        # overwrite this in each trader
        return {}  

class AcoTrader(ProductTrader):
    """
    ACO (ASH_COATED_OSMIUM): is a bit more volatile, although one may speculate that its apparent unpredictability may follow a hidden pattern.
    ACO become stable stationary when normalized by wall_mid, so treat wall_mid as fair.
    ADF p-value = 0.0000 → stationary/mean-reverting → good for market making
    how can we exploit this?
    """
    def __init__(self, state, prints, new_trader_data):
        super().__init__("ASH_COATED_OSMIUM", state, prints, new_trader_data, product_group="ACO")

    def get_orders(self):

        if self.wall_mid is None:
            return {self.name: self.orders}

        ##########################################################
        ####### 1. TAKING
        ##########################################################
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= self.wall_mid - 1:
                self.bid(sp, sv, logging=False)
            elif sp <= self.wall_mid and self.initial_position < 0:
                volume = min(sv, abs(self.initial_position))
                self.bid(sp, volume, logging=False)

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= self.wall_mid + 1:
                self.ask(bp, bv, logging=False)
            elif bp >= self.wall_mid and self.initial_position > 0:
                volume = min(bv, self.initial_position)
                self.ask(bp, volume, logging=False)

        ###########################################################
        ####### 2. MAKING
        ###########################################################
        bid_price = int(self.bid_wall + 1)  # base case
        ask_price = int(self.ask_wall - 1)  # base case

        # OVERBIDDING: overbid best bid that is still under the mid wall
        for bp, bv in self.mkt_buy_orders.items():
            overbidding_price = bp + 1
            if bv > 1 and overbidding_price < self.wall_mid:
                bid_price = max(bid_price, overbidding_price)
                break
            elif bp < self.wall_mid:
                bid_price = max(bid_price, bp)
                break

        # UNDERBIDDING: underbid best ask that is still over the mid wall
        for sp, sv in self.mkt_sell_orders.items():
            underbidding_price = sp - 1
            if sv > 1 and underbidding_price > self.wall_mid:
                ask_price = min(ask_price, underbidding_price)
                break
            elif sp > self.wall_mid:
                ask_price = min(ask_price, sp)
                break

        # --- Inventory-based skew (1 tick) ---
        # ACO spreads are wide; we can use 1-tick inventory control without sacrificing much edge.
        # To avoid needless edge loss, apply price skew only on the inventory-worsening side and
        # complement with size tilt.
        intended_pos = self.initial_position
        if self.position_limit > 0:
            submitted_buy = (self.position_limit - self.initial_position) - self.max_allowed_buy_volume
            submitted_sell = (self.position_limit + self.initial_position) - self.max_allowed_sell_volume
            intended_pos = self.initial_position + submitted_buy - submitted_sell

        inv_ratio = 0.0
        if self.position_limit > 0:
            inv_ratio = intended_pos / self.position_limit

        INV_SKEW_THRESHOLD = 0.3
        if inv_ratio >= INV_SKEW_THRESHOLD:
            inv_skew = -1  # long → discourage buying
        elif inv_ratio <= -INV_SKEW_THRESHOLD:
            inv_skew = 1   # short → discourage selling
        else:
            inv_skew = 0

        if inv_skew < 0:
            bid_price -= 1
        elif inv_skew > 0:
            ask_price += 1

        bid_price = min(bid_price, int(self.wall_mid) - 1)
        ask_price = max(ask_price, int(self.wall_mid) + 1)

        buy_cap = max(0, int(self.max_allowed_buy_volume))
        sell_cap = max(0, int(self.max_allowed_sell_volume))

        buy_tilt = max(0.3, 1.0 - max(0, inv_ratio))
        sell_tilt = max(0.3, 1.0 + min(0, inv_ratio))
        buy_vol = int(round(buy_cap * buy_tilt))
        sell_vol = int(round(sell_cap * sell_tilt))

        if buy_vol > 0:
            self.bid(bid_price, buy_vol)
        if sell_vol > 0:
            self.ask(ask_price, sell_vol)

        return {self.name: self.orders}


class IprTrader(ProductTrader):
    """
    IPR (INTARIAN_PEPPER_ROOT): consistent upward drift ~1000/day.
    Drift per timestamp step ≈ 0.1 tick, far exceeding any spread cost (~7-9 ticks).
    Strategy: maximize long position ASAP, hold for drift gains.
    Holding 80 units for a full day captures ~80,000 in mark-to-market.
    """
    def __init__(self, state, prints, new_trader_data):
        super().__init__("INTARIAN_PEPPER_ROOT", state, prints, new_trader_data, product_group="IPR")

    def get_orders(self):
        if self.wall_mid is None:
            return {self.name: self.orders}

        ##########################################################
        # 1. TAKE all available asks (spread cost is negligible vs drift)
        ##########################################################
        for sp, sv in self.mkt_sell_orders.items():
            self.bid(sp, sv, logging=False)

        ##########################################################
        # 2. POST aggressive bid for remaining buy capacity
        ##########################################################
        buy_cap = max(0, int(self.max_allowed_buy_volume))
        if buy_cap > 0:
            bid_price = int(self.wall_mid)
            if self.best_bid is not None:
                bid_price = max(bid_price, int(self.best_bid) + 1)
            bid_price = min(bid_price, int(self.wall_mid) + 1)
            self.bid(bid_price, buy_cap)

        return {self.name: self.orders}


# Hydrogel Pack
class HpackTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(HPACK_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):

        if self.wall_mid is not None:

            ##########################################################
            ####### 1. TAKING
            ##########################################################
            for sp, sv in self.mkt_sell_orders.items():
                if sp <= self.wall_mid - 1:
                    self.bid(sp, sv, logging=False)
                elif sp <= self.wall_mid and self.initial_position < 0:
                        volume = min(sv,  abs(self.initial_position))
                        self.bid(sp, volume, logging=False)

            for bp, bv in self.mkt_buy_orders.items():
                if bp >= self.wall_mid + 1:
                    self.ask(bp, bv, logging=False)
                elif bp >= self.wall_mid and self.initial_position > 0:
                        volume = min(bv,  self.initial_position)
                        self.ask(bp, volume, logging=False)

            ###########################################################
            ####### 2. MAKING
            ###########################################################
            bid_price = int(self.bid_wall + 1) # base case
            ask_price = int(self.ask_wall - 1) # base case

            # OVERBIDDING: overbid best bid that is still under the mid wall
            for bp, bv in self.mkt_buy_orders.items():
                overbidding_price = bp + 1
                if bv > 1 and overbidding_price < self.wall_mid:
                    bid_price = max(bid_price, overbidding_price)
                    break
                elif bp < self.wall_mid:
                    bid_price = max(bid_price, bp)
                    break

            # UNDERBIDDING: underbid best ask that is still over the mid wall
            for sp, sv in self.mkt_sell_orders.items():
                underbidding_price = sp - 1
                if sv > 1 and underbidding_price > self.wall_mid:
                    ask_price = min(ask_price, underbidding_price)
                    break
                elif sp > self.wall_mid:
                    ask_price = min(ask_price, sp)
                    break

            # POST ORDERS
            self.bid(bid_price, self.max_allowed_buy_volume)
            self.ask(ask_price, self.max_allowed_sell_volume)


        return {self.name: self.orders}



# Velvetfruit Extract + 10 Vouchers — IV-SCALPING strategy
#
# Senior team's Round-3 framework (Frankfurt Hedgehogs), recalibrated for VEV.
# See "VEV / VOUCHERS — DATA CHARACTERISTICS" up top for the supporting numbers.
#
#   1. ACTIVE vouchers (5100-5500): IV scalping.
#        - online quadratic smile fit each tick (IV vs moneyness)
#        - theo_diff       = wall_mid - BS_theo(smile_iv)
#        - mean_theo_diff  = EMA(theo_diff, 20)              # absorbs bias
#        - signal          = theo_diff - mean_theo_diff       # the alpha
#        - activity        = EMA(|signal|, 100)               # gating meter
#        - trade only if activity >= IV_SCALPING_THR
#        - SELL when (best_bid - theo - mean_theo_diff) >= THR_OPEN
#          BUY  when (best_ask - theo - mean_theo_diff) <= -THR_OPEN
#        - flatten gracefully when activity drops below the gate.
#   2. DEEP-ITM-ish proxy (5000): mean-reversion in price space, combined with
#      the underlying's ema_o_dev (senior used 9500 the same way).
#   3. DEEP ITM (4000, 4500): SKIP — pinned at intrinsic ± 10, no extrinsic
#      edge, unhedgeable cost.  Same for DEAD strikes (6000, 6500).
#   4. UNDERLYING (VE): standalone EMA-MR taker; alpha is small (ac1≈-0.04).
#      No explicit delta hedging — IV scalping holds positions in short
#      bursts and the senior team explicitly avoided the bid-ask cost.
class VeoTrader:
    def __init__(self, state, prints, new_trader_data):

        # Carrier set for the smile fit + indicators.  Includes both
        # scalping and MR strikes (only those we trade).
        self.options        = [ProductTrader(s, state, prints, new_trader_data, product_group='VEO')
                               for s in VEV_SCALPING_STRIKES + VEV_MR_STRIKES]
        # Passive smile-only carriers: never ordered, only provide IV data points.
        self.smile_carriers = [ProductTrader(s, state, prints, new_trader_data, product_group='VEO')
                               for s in VEV_SMILE_ONLY]
        self.underlying = ProductTrader(VE_UNDERLYING_SYMBOL, state, prints, new_trader_data, product_group='VEO')

        self.state = state
        self.last_traderData = self.underlying.last_traderData
        self.new_trader_data = new_trader_data

        self.indicators = self.calculate_indicators()


    def calculate_ema(self, td_key, window, value):
        # Reset to current value at the start of each new day (timestamp == 0)
        # or on cold start (key unseen). This prevents the large warmup spike
        # caused by EMA initialised to 0 while the true mean is 1-3 points away,
        # and eliminates cross-day carryover bias.
        if self.state.timestamp == 0 or td_key not in self.last_traderData:
            self.new_trader_data[td_key] = value
            return value
        old_mean = self.last_traderData[td_key]
        alpha = 2/(window+1)
        new_mean = alpha * value + (1 - alpha) * old_mean
        self.new_trader_data[td_key] = new_mean
        return new_mean


    def bs_call(self, S, K, TTE, s, r=0):
        d1 = (math.log(S/K) + (r + 0.5 * s**2) * TTE) / (s * TTE**0.5)
        d2 = d1 - s * TTE**0.5
        return S * _N.cdf(d1) - K * math.exp(-r * TTE) * _N.cdf(d2), _N.cdf(d1)


    def bs_vega(self, S, K, TTE, s, r=0):
        if s <= 0 or TTE <= 0: return 0.0
        d1 = (math.log(S/K) + (r + 0.5 * s**2) * TTE) / (s * TTE**0.5)
        return S * _N.pdf(d1) * TTE**0.5


    def implied_vol(self, S, K, TTE, mkt_px):
        # Bisection on a monotonically-increasing call price in vol.
        # Returns None if mkt_px <= intrinsic (no positive-vol root).
        intrinsic = max(0.0, S - K)
        if mkt_px <= intrinsic + 1e-6: return None
        lo, hi = 1e-4, 5.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            px, _delta = self.bs_call(S, K, TTE, mid)
            if px > mkt_px: hi = mid
            else: lo = mid
            if hi - lo < 1e-5: break
        return 0.5 * (lo + hi)


    def fit_smile(self, ms, ivs):
        # Quadratic fit IV ~ a*m^2 + b*m + c.
        # Guards:
        #   1. Need at least VEV_SMILE_MIN_POINTS clean IVs.
        #   2. Moneyness range must be wide enough (VEV_SMILE_MIN_M_SPREAD) —
        #      a narrow range makes the parabola degenerate / ill-conditioned.
        # After fitting, EMA-smooth the coefficients across ticks so a single
        # noisy quote cannot spike all theos simultaneously.
        if len(ms) < VEV_SMILE_MIN_POINTS:
            return self._ema_smile(VEV_SMILE_COEFFS_FALLBACK)
        if max(ms) - min(ms) < VEV_SMILE_MIN_M_SPREAD:
            return self._ema_smile(VEV_SMILE_COEFFS_FALLBACK)
        try:
            a, b, c = np.polyfit(ms, ivs, 2)
            # Basic sanity: curvature must be positive, ATM IV in [0.05, 1.0]
            if a < 0 or not (0.05 <= c <= 1.0):
                return self._ema_smile(VEV_SMILE_COEFFS_FALLBACK)
            return self._ema_smile([float(a), float(b), float(c)])
        except Exception:
            return self._ema_smile(VEV_SMILE_COEFFS_FALLBACK)

    def _ema_smile(self, raw_coeffs):
        # Apply EMA smoothing to each smile coefficient independently.
        a = self.calculate_ema('smile_a', VEV_SMILE_EMA_WINDOW, raw_coeffs[0])
        b = self.calculate_ema('smile_b', VEV_SMILE_EMA_WINDOW, raw_coeffs[1])
        c = self.calculate_ema('smile_c', VEV_SMILE_EMA_WINDOW, raw_coeffs[2])
        return (a, b, c)


    def get_option_values(self, S, K, TTE, smile):
        m = math.log(K/S) / TTE**0.5
        a, b, c = smile
        iv = max(1e-3, a*m*m + b*m + c)
        theo, delta = self.bs_call(S, K, TTE, iv)
        vega = self.bs_vega(S, K, TTE, iv)
        return theo, delta, vega, iv


    def calculate_indicators(self):

        indicators = {
            'S': None, 'TTE': None, 'smile': None,
            'ema_u_dev': None,           # VE underlying MR signal (window 20)
            'ema_o_dev': None,           # VE EMA used by deep-ITM-option MR (window 30)
            'theo_diffs': {}, 'mean_theo_diffs': {}, 'signals': {}, 'activities': {},
            'deltas': {}, 'vegas': {}, 'ivs': {},
        }

        if self.underlying.wall_mid is None:
            return indicators

        S = self.underlying.wall_mid
        TTE = 1 - (DAYS_PER_YEAR - 8 + DAY + self.state.timestamp // 100 / 10_000) / DAYS_PER_YEAR
        TTE = max(1e-6, TTE)
        sqrtT = TTE ** 0.5

        indicators['S'] = S
        indicators['TTE'] = TTE

        # Two EMAs of VE wall_mid: 20-tick (underlying MR) and 30-tick (option MR).
        ema_u = self.calculate_ema('ema_u', ve_underlying_mean_reversion_window, S)
        indicators['ema_u_dev'] = S - ema_u
        ema_o = self.calculate_ema('ema_o', options_mean_reversion_window, S)
        indicators['ema_o_dev'] = S - ema_o

        # 1. Online smile fit — collect IVs from both trading options AND
        #    passive smile carriers (5300, 5500) for wider moneyness coverage.
        ms_clean, ivs_clean = [], []
        for option in self.options + self.smile_carriers:
            if option.wall_mid is None: continue
            K = int(option.name.split('_')[-1])
            iv = self.implied_vol(S, K, TTE, option.wall_mid)
            if iv is None: continue
            if option.name not in VEV_SMILE_ONLY:           # only log traded strikes
                indicators['ivs'][option.name] = iv
            ms_clean.append(math.log(K/S) / sqrtT)
            ivs_clean.append(iv)

        smile = self.fit_smile(ms_clean, ivs_clean)
        indicators['smile'] = smile

        # 2. Per-option theo, theo_diff, EMAs, signal, activity.
        for option in self.options:
            if option.wall_mid is None: continue
            K = int(option.name.split('_')[-1])
            theo, delta, vega, _iv = self.get_option_values(S, K, TTE, smile)
            theo_diff = option.wall_mid - theo

            mean_theo_diff = self.calculate_ema(f'{option.name}_theo_diff', THEO_NORM_WINDOW, theo_diff)
            activity       = self.calculate_ema(f'{option.name}_activity',  IV_SCALPING_WINDOW, abs(theo_diff - mean_theo_diff))

            indicators['theo_diffs'][option.name]      = theo_diff
            indicators['mean_theo_diffs'][option.name] = mean_theo_diff
            indicators['signals'][option.name]         = theo_diff - mean_theo_diff
            indicators['activities'][option.name]      = activity
            indicators['deltas'][option.name]          = delta
            indicators['vegas'][option.name]           = vega

        return indicators


    def get_iv_scalping_orders(self, options):

        # Senior pattern verbatim, in PRICE space:
        #   edge_sell = best_bid - theo - mean_theo_diff   ;   open if >= THR_OPEN
        #   edge_buy  = best_ask - theo - mean_theo_diff   ;   open if <= -THR_OPEN
        # Use the equivalent algebra `current_theo_diff - wall_mid + best_bid - mean_theo_diff`
        # so a single expression handles both sides cleanly.

        out = {}

        for option in options:
            sym = option.name
            if sym not in self.indicators['mean_theo_diffs']:
                out[sym] = option.orders
                continue

            activity       = self.indicators['activities'].get(sym, 0.0)
            current_td     = self.indicators['theo_diffs'][sym]
            mean_td        = self.indicators['mean_theo_diffs'][sym]
            vega           = self.indicators['vegas'].get(sym, 0.0)
            low_vega_adj   = LOW_VEGA_THR_ADJ if vega <= 1 else 0.0

            if activity >= IV_SCALPING_THR:

                # SELL OPEN (and immediately add a CLOSE-longs leg if applicable)
                if (option.best_bid is not None
                        and current_td - option.wall_mid + option.best_bid - mean_td >= (THR_OPEN + low_vega_adj)
                        and option.max_allowed_sell_volume > 0):
                    option.ask(option.best_bid, option.max_allowed_sell_volume)

                if (option.best_bid is not None
                        and current_td - option.wall_mid + option.best_bid - mean_td >= THR_CLOSE
                        and option.initial_position > 0):
                    option.ask(option.best_bid, option.initial_position)

                # BUY OPEN (mirror)
                elif (option.best_ask is not None
                        and current_td - option.wall_mid + option.best_ask - mean_td <= -(THR_OPEN + low_vega_adj)
                        and option.max_allowed_buy_volume > 0):
                    option.bid(option.best_ask, option.max_allowed_buy_volume)

                if (option.best_ask is not None
                        and current_td - option.wall_mid + option.best_ask - mean_td <= -THR_CLOSE
                        and option.initial_position < 0):
                    option.bid(option.best_ask, -option.initial_position)

            else:
                # Activity dropped below gate: flatten any open position passively.
                if option.initial_position > 0 and option.best_bid is not None:
                    option.ask(option.best_bid, option.initial_position)
                elif option.initial_position < 0 and option.best_ask is not None:
                    option.bid(option.best_ask, -option.initial_position)

            out[sym] = option.orders

        return out


    def get_mr_orders(self, options):

        # Deep-ITM-ish MR (senior used 9500): combine the underlying ema_o_dev
        # with the option's own iv_deviation (theo_diff - mean_theo_diff) so we
        # MR more confidently when both signals point the same way.

        out = {}

        ema_o_dev = self.indicators.get('ema_o_dev')
        if ema_o_dev is None:
            return out

        for option in options:
            sym = option.name
            if sym not in self.indicators['theo_diffs']:
                continue

            iv_deviation = self.indicators['theo_diffs'][sym] - self.indicators['mean_theo_diffs'][sym]
            current_dev  = ema_o_dev + iv_deviation

            if current_dev > options_mean_reversion_thr and option.max_allowed_sell_volume > 0 and option.best_bid is not None:
                option.ask(option.best_bid, option.max_allowed_sell_volume)
            elif current_dev < -options_mean_reversion_thr and option.max_allowed_buy_volume > 0 and option.best_ask is not None:
                option.bid(option.best_ask, option.max_allowed_buy_volume)

            out[sym] = option.orders

        return out


    def get_option_orders(self):

        # Warmup: every EMA must have time to settle before we trust signals.
        warmup = max(THEO_NORM_WINDOW, IV_SCALPING_WINDOW,
                     ve_underlying_mean_reversion_window, options_mean_reversion_window)
        if self.state.timestamp / 100 < warmup:
            return {}

        scalping_opts = [o for o in self.options if o.name in VEV_SCALPING_STRIKES]
        mr_opts       = [o for o in self.options if o.name in VEV_MR_STRIKES]

        return {
            **self.get_iv_scalping_orders(scalping_opts),
            **self.get_mr_orders(mr_opts),
        }


    def get_underlying_orders(self):

        # Standalone MR taker on VE. No delta hedging — IV scalping turns
        # over fast and the senior team explicitly avoided hedging because
        # of the bid-ask spread cost.

        out = {self.underlying.name: self.underlying.orders}
        if self.underlying.wall_mid is None: return out
        if self.state.timestamp / 100 < ve_underlying_mean_reversion_window: return out

        ema_u_dev = self.indicators.get('ema_u_dev')
        if ema_u_dev is None: return out

        if (ema_u_dev > ve_underlying_mean_reversion_thr
                and self.underlying.max_allowed_sell_volume > 0
                and self.underlying.bid_wall is not None):
            self.underlying.ask(self.underlying.bid_wall + 1, self.underlying.max_allowed_sell_volume)

        elif (ema_u_dev < -ve_underlying_mean_reversion_thr
                and self.underlying.max_allowed_buy_volume > 0
                and self.underlying.ask_wall is not None):
            self.underlying.bid(self.underlying.ask_wall - 1, self.underlying.max_allowed_buy_volume)

        return out


    def get_orders(self):

        # Order important: option scalping first (it sets initial_position
        # context for the close-leg logic), then the standalone VE MR.
        orders = {
            **self.get_option_orders(),
            **self.get_underlying_orders(),
        }

        return orders



class Trader:

    def run(self, state: TradingState):
        result:dict[str,list[Order]] = {}
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position
            },
        }

        def export(prints):
            try: print(json.dumps(prints))
            except: pass


        product_traders = {
            HPACK_SYMBOL: HpackTrader,
            VE_UNDERLYING_SYMBOL: VeoTrader,
            # Local products from earlier rounds
            # "ASH_COATED_OSMIUM": AcoTrader,
            # "INTARIAN_PEPPER_ROOT": IprTrader,
        }

        result, conversions = {}, 0
        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:

                try:
                    trader = product_trader(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except: pass


        try: final_trader_data = json.dumps(new_trader_data)
        except: final_trader_data = ''


        export(prints)
        return result, conversions, final_trader_data