from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json

"""
When players upload their algorithms on the Prosperity platform, the algorithm is tested for 1000 iterations using data from a sample day 
(different than the actual day that will be used for the challenge). 
After the run a log file is provided which can aid players in debugging their algorithms. 
To aid debugging, the log file also contains the output of any print statements that players put within the run method of their trading class.

"""

ETF_BASKET_SYMBOLS = ['PICNIC_BASKET1', 'PICNIC_BASKET2']
ETF_CONSTITUENT_SYMBOLS = ['CROISSANTS', 'JAMS', 'DJEMBES']

STATIC_SYMBOL = 'RAINFOREST_RESIN'
DYNAMIC_SYMBOL = 'KELP'
INK_SYMBOL = 'SQUID_INK'

OPTION_UNDERLYING_SYMBOL = 'VOLCANIC_ROCK'

COMMODITY_SYMBOL = 'MAGNIFICENT_MACARONS'

OPTION_SYMBOLS = [
    'VOLCANIC_ROCK_VOUCHER_9500',
    'VOLCANIC_ROCK_VOUCHER_9750',
    'VOLCANIC_ROCK_VOUCHER_10000',
    'VOLCANIC_ROCK_VOUCHER_10250',
    'VOLCANIC_ROCK_VOUCHER_10500'
    ]

POS_LIMITS = {
    STATIC_SYMBOL: 80, 
    DYNAMIC_SYMBOL: 80,
    INK_SYMBOL: 50,
    ETF_BASKET_SYMBOLS[0]: 60,
    ETF_BASKET_SYMBOLS[1]: 100,
    ETF_CONSTITUENT_SYMBOLS[0]: 250,
    ETF_CONSTITUENT_SYMBOLS[1]: 350,
    ETF_CONSTITUENT_SYMBOLS[2]: 60,

    OPTION_UNDERLYING_SYMBOL: 400,
    **{os: 200 for os in OPTION_SYMBOLS},

    COMMODITY_SYMBOL: 75,

    # Local products
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

CONVERSION_LIMIT = 10

LONG, NEUTRAL, SHORT = 1, 0, -1


INFORMED_TRADER_ID = 'Olivia'

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


class Trader:
    def bid(self):
        return 500
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position,
            },
        }

        def export(p):
            try:
                print(json.dumps(p))
            except Exception:
                pass

        product_traders = {
            "ASH_COATED_OSMIUM": AcoTrader,
            "INTARIAN_PEPPER_ROOT": IprTrader,
        }

        conversions = 0
        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = product_trader(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except Exception:
                    pass

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ""

        export(prints)
        return result, conversions, final_trader_data