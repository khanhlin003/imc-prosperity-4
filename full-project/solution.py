import numpy as np
from scipy.stats import norm

S0, sig, r = 50.0, 2.51, 0.0
T2 = (2*5)/252   # 10/252 years
T3 = (3*5)/252   # 15/252 years

def bs_d1d2(S, K, T):
    sqT = np.sqrt(T)
    d1  = (np.log(S / K) + 0.5 * sig**2 * T) / (sig * sqT)
    return d1, d1 - sig * sqT

def bs_call(S, K, T):
    d1, d2 = bs_d1d2(S, K, T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def bs_put(S, K, T):
    d1, d2 = bs_d1d2(S, K, T)
    return K * norm.cdf(-d2) - S * norm.cdf(-d1)

print("=== TIME CHECK ===")
print(f"T2 = {T2:.6f} yr  ({T2*252:.1f} trading days, {T2*252*4:.0f} steps)")
print(f"T3 = {T3:.6f} yr  ({T3*252:.1f} trading days, {T3*252*4:.0f} steps)")
print(f"σ√T2 = {sig*np.sqrt(T2):.4f}")
print(f"σ√T3 = {sig*np.sqrt(T3):.4f}")

print("\n=== VANILLA FAIR VALUES vs MARKET ===")
specs = [
    ("AC_50_P",   "put",  50, T3, 12.00, 12.05),
    ("AC_50_C",   "call", 50, T3, 12.00, 12.05),
    ("AC_35_P",   "put",  35, T3,  4.33,  4.35),
    ("AC_40_P",   "put",  40, T3,  6.50,  6.55),
    ("AC_45_P",   "put",  45, T3,  9.05,  9.10),
    ("AC_60_C",   "call", 60, T3,  8.80,  8.85),
    ("AC_50_P_2", "put",  50, T2,  9.70,  9.75),
    ("AC_50_C_2", "call", 50, T2,  9.70,  9.75),
]
for name, typ, K, T, bid, ask in specs:
    fv = bs_call(S0, K, T) if typ=="call" else bs_put(S0, K, T)
    e_buy  = fv - ask
    e_sell = bid - fv
    best = "BUY" if e_buy >= 0.05 else ("SELL" if e_sell >= 0.05 else "PASS")
    print(f"  {name:12s}  fair={fv:7.4f}  bid={bid:.3f}  ask={ask:.3f}  "
          f"edge_buy={e_buy:+.4f}  edge_sell={e_sell:+.4f}  → {best}")

print("\n=== CHOOSER ===")
fv_co = bs_call(S0, 50, T3) + bs_put(S0, 50, T2)
print(f"  V_chooser = C(50,T3) + P(50,T2) = {bs_call(S0,50,T3):.4f} + {bs_put(S0,50,T2):.4f} = {fv_co:.4f}")
print(f"  Market bid=22.20, ask=22.30")
print(f"  edge_buy = {fv_co-22.30:+.4f}  edge_sell = {22.20-fv_co:+.4f}")

print("\n=== BINARY PUT (AC_40_BP, payout=10) ===")
_, d2_bp = bs_d1d2(S0, 40, T3)
fv_bp = 10 * norm.cdf(-d2_bp)
print(f"  d2 = {d2_bp:.4f},  N(-d2) = {norm.cdf(-d2_bp):.4f}")
print(f"  Fair = 10 × {norm.cdf(-d2_bp):.4f} = {fv_bp:.4f}")
print(f"  Market bid=5.00, ask=5.10")
print(f"  edge_buy = {fv_bp-5.10:+.4f}  edge_sell = {5.00-fv_bp:+.4f}")

print("\n=== KO PUT MC (1M paths) ===")
np.random.seed(42)
n_steps = int(round(T3 * 252 * 4))
dt      = T3 / n_steps
Z = np.random.standard_normal((1_000_000, n_steps))
paths = np.exp(np.log(S0) + np.cumsum(-0.5*sig**2*dt + sig*np.sqrt(dt)*Z, axis=1))
alive   = paths.min(axis=1) >= 35
S_T     = paths[:, -1]
payoffs = np.maximum(45 - S_T, 0.0) * alive
fv_ko   = payoffs.mean()
ci_ko   = 1.96*payoffs.std()/np.sqrt(1_000_000)
ko_prob = (~alive).mean()
print(f"  Knockout probability = {ko_prob:.4f} ({ko_prob:.2%})")
print(f"  Vanilla put (K=45)   = {bs_put(S0, 45, T3):.4f}")
print(f"  KO fair value (MC)   = {fv_ko:.4f} ± {ci_ko:.4f}")
print(f"  Market bid=0.15, ask=0.175")
print(f"  edge_buy = {fv_ko-0.175:+.4f}  edge_sell = {0.15-fv_ko:+.4f}")

print("\n=== DELTA CHECK (approx) ===")
dS = 0.01
trades = [
    ("AC_50_P_2", "put",  50, T2, "BUY",  50),
    ("AC_50_C_2", "call", 50, T2, "BUY",  50),
    ("AC_50_CO",  None,   50, None,"SELL", 50),
    ("AC_40_BP",  None,   40, T3, "SELL", 50),
]
# Add KO depending on direction
ko_action = "BUY" if fv_ko - 0.175 >= 0.05 else ("SELL" if 0.15 - fv_ko >= 0.05 else "PASS")
if ko_action != "PASS":
    trades.append(("AC_45_KO", "ko", 45, T3, ko_action, 500))

net_delta = 0.0
for name, typ, K, T_opt, action, qty in trades:
    T_use = T_opt if T_opt else T3
    if typ == "call":
        d1,_ = bs_d1d2(S0, K, T_use); dlt = norm.cdf(d1)
    elif typ == "put":
        d1,_ = bs_d1d2(S0, K, T_use); dlt = norm.cdf(d1) - 1
    elif typ is None and name == "AC_50_CO":
        d1_t3,_ = bs_d1d2(S0, 50, T3); d1_t2,_ = bs_d1d2(S0, 50, T2)
        dlt = norm.cdf(d1_t3) + (norm.cdf(d1_t2)-1)  # dC/dS + dP/dS
    elif typ is None:  # binary put
        _, d2_ = bs_d1d2(S0, K, T_use)
        dlt = -10 * norm.pdf(d2_) / (S0 * sig * np.sqrt(T_use))
    else:  # KO
        p_alive = 1 - ko_prob
        d1,_ = bs_d1d2(S0, K, T_use); dlt = (norm.cdf(d1)-1) * p_alive
    sign = +1 if action == "BUY" else -1
    contrib = sign * qty * dlt
    net_delta += contrib
    print(f"  {action} {qty:3d} {name:12s}  Δ/contract={dlt:+.4f}  contrib={contrib:+.2f}")

print(f"\n  Net Δ ≈ {net_delta:+.2f}")
print(f"  KO action: {ko_action}")
if abs(net_delta) > 10:
    ac_side = "BUY" if net_delta < 0 else "SELL"
    ac_qty = min(200, int(abs(round(net_delta))))
    print(f"  → Recommend: {ac_side} {ac_qty} AC at ~50 to partially hedge")
