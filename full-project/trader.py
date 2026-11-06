from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json

"""
When players upload their algorithms on the Prosperity platform, the algorithm is tested for 1000 iterations using data from a sample day 
(different than the actual day that will be used for the challenge). 
After the run a log file is provided which can aid players in debugging their algorithms. 
To aid debugging, the log file also contains the output of any print statements that players put within the run method of their trading class.

"""


class Trader:

    # ── Constants ──────────────────────────────────────────────────────────
    ACO  = "ASH_COATED_OSMIUM"
    IPR  = "INTARIAN_PEPPER_ROOT"

    ACO_FAIR_VALUE   = 10_000
    ACO_SPREAD       = 7        # quote fair ± 7, captures ~half the avg spread of 16
    ACO_MAX_POSITION = 80       # adjust to actual position limit

    IPR_BASE         = 10_000
    IPR_DRIFT_DAILY  = 1_000    # +1000 per day
    IPR_DRIFT_INTRA  = 0.001    # +0.001 per timestamp unit
    IPR_MAX_POSITION = 80       # adjust to actual position limit
    IPR_ENTRY_THRESH = -50      # only buy if price is this far BELOW fair value
    IPR_EXIT_THRESH  = 200      # trim if price runs this far ABOVE fair value

    # ── Fair Value ─────────────────────────────────────────────────────────
    def aco_fair_value(self) -> float:
        return self.ACO_FAIR_VALUE

    # def ipr_fair_value(self, day: int, timestamp: int) -> float:
    #     base     = self.IPR_BASE + (day + 2) * self.IPR_DRIFT_DAILY
    #     intraday = self.IPR_DRIFT_INTRA * timestamp
    #     return base + intraday
    def ipr_fair_value(self, timestamp: int) -> float:
        base = self.IPR_BASE

        # intraday drift (smooth upward curve)
        t = timestamp / 1_000_000
        drift = self.IPR_DRIFT_INTRA * timestamp

        return base + drift

    def ipr_regime(self, timestamp: int) -> float:
        if timestamp < 250_000:
            return -25   # undervalued early bias
        elif timestamp < 750_000:
            return 0
        else:
            return 25    # overvaluation late bias

    # ── Order Helpers ──────────────────────────────────────────────────────
    def best_bid(self, od: OrderDepth):
        return max(od.buy_orders.keys())  if od.buy_orders  else None

    def best_ask(self, od: OrderDepth):
        return min(od.sell_orders.keys()) if od.sell_orders else None

    def mid(self, od: OrderDepth):
        b, a = self.best_bid(od), self.best_ask(od)
        if b and a:
            return (b + a) / 2
        return None

    # ── ACO: Market Making ─────────────────────────────────────────────────
    """
    Explaination for ACO product:
    Competition Description: ASH_COATED_OSMIUM is rumored to be a bit more volatile, although one may speculate that its apparent unpredictability may follow a hidden pattern.
    ASH_COATED_OSMIUM: ADF p-value = 0.0000 -> stationary
    accumulated pnl for aco is from 0 to 2k1 
    """
    def trade_aco(self, state: TradingState) -> List[Order]:
        orders   = []
        od       = state.order_depths[self.ACO]
        position = state.position.get(self.ACO, 0)
        fv       = self.aco_fair_value()
        lim      = self.ACO_MAX_POSITION

        # Inventory skew — shift quotes to push position back toward 0
        # skew ranges from -1 (max short) to +1 (max long)
        skew      = position / lim
        half      = self.ACO_SPREAD
        bid_price = round(fv - half - skew * half)
        ask_price = round(fv + half - skew * half)

        # Size — don't exceed position limit
        buy_capacity  =  lim - position   # how many more we can buy
        sell_capacity =  lim + position   # how many more we can sell
        quote_size    = max(1, lim // 5)  # quote ~20% of limit at a time

        buy_qty  = min(quote_size, buy_capacity)
        sell_qty = min(quote_size, sell_capacity)

        # Also take any existing orders that are mispriced
        # (someone selling below fair value = free money)
        best_a = self.best_ask(od)
        best_b = self.best_bid(od)

        if best_a and best_a < fv - 1 and buy_capacity > 0:
            take_qty = min(-od.sell_orders[best_a], buy_capacity)
            orders.append(Order(self.ACO, best_a, take_qty))
            print(f"TAKING LIQUIDITY: Bought {take_qty} ACO at {best_a} (Fair Value: {fv})")
            buy_capacity -= take_qty

        if best_b and best_b > fv + 1 and sell_capacity > 0:
            take_qty = min(od.buy_orders[best_b], sell_capacity)
            orders.append(Order(self.ACO, best_b, -take_qty))
            print(f"TAKING LIQUIDITY: Sold {take_qty} ACO at {best_b} (Fair Value: {fv})")
            sell_capacity -= take_qty

        # Post passive quotes
        if buy_qty > 0:
            orders.append(Order(self.ACO, bid_price,  buy_qty))
        if sell_qty > 0:
            orders.append(Order(self.ACO, ask_price, -sell_qty))

        return orders
    

    # ── IPR: Trend Following ───────────────────────────────────────────────
    """
    Explaination for IPR product:
    Competition Description: The value of INTARIAN_PEPPER_ROOT is quite steady, but keep in mind that it's a hardy, slow-growing root.
    INTARIAN_PEPPER_ROOT: ADF p-value = 0.9449 -> trending
    accumulated pnl for ipr is from -181 to 2k8 over 100 
    """
    def trade_ipr(self, state: TradingState) -> List[Order]:
            orders = []
            od = state.order_depths[self.IPR]
            pos = state.position.get(self.IPR, 0)
            lim = self.IPR_MAX_POSITION
            
            # 1. Detect the Trend (Simple Momentum)
            # Use a longer window for the trend (e.g., last 1000 timestamps)
            mid_price = self.mid(od)
            if mid_price is None: return []

            # 2. Buy on Pullbacks (Scaling In)
            # Instead of 'if deviation < -15', check if the price is 
            # below the recent trend average
            recent_average = 12_000 # Replace this with your actual moving average logic
            
            # Scale in: only buy if we are not at limit
            if mid_price < recent_average and pos < lim:
                buy_qty = min(20, lim - pos) # Scale in 20 units at a time
                best_a = self.best_ask(od)
                if best_a:
                    orders.append(Order(self.IPR, best_a, buy_qty))
                    print(f"TREND-FOLLOWING: Buying pullback {buy_qty} @ {best_a}")

            # 3. Hold for the Trend (NO Selling into strength)
            # Only sell if the trend reverses (mid_price drops below a hard floor)
            floor = 11_000 
            if mid_price < floor and pos > 0:
                best_b = self.best_bid(od)
                if best_b:
                    orders.append(Order(self.IPR, best_b, -pos)) # Close out
                    print(f"TREND-REVERSAL: Exiting full position @ {best_b}")

            return orders

    # ── Main Entry Point ───────────────────────────────────────────────────
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        # trader_state = self.load_state(state.traderData)

        if self.ACO in state.order_depths:
            result[self.ACO] = self.trade_aco(state)
            
        if self.IPR in state.order_depths:
            result[self.IPR] = self.trade_ipr(state)

        return result, 0, json.dumps({})