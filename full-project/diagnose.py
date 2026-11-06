import pandas as pd, numpy as np, math
from scipy.stats import norm

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0: return max(S-K, 0)
    d1 = (math.log(S/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*norm.cdf(d2)

smile_by_day = {0: [0.117,-0.008,0.304], 1: [0.113,-0.004,0.316], 2: [0.097,0.004,0.355]}

for DAY in [0, 1, 2]:
    df = pd.read_csv(f"data/round3/prices_round_3_day_{DAY}.csv", sep=";")
    ve = df[df['product']=='VELVETFRUIT_EXTRACT'][['timestamp','mid_price']].rename(columns={'mid_price':'S'})
    smile = smile_by_day[DAY]

    print(f"\n=== DAY {DAY} ===")
    for sym, K in [('VEV_5100', 5100), ('VEV_5200', 5200), ('VEV_5400', 5400)]:
        vev = df[df['product']==sym][['timestamp','mid_price','bid_price_1','ask_price_1']].rename(columns={'mid_price':'mkt'})
        merged = ve.merge(vev, on='timestamp')
        if merged.empty:
            print(f"  {sym}: no data")
            continue

        theos = []
        spreads = []
        for _, r in merged.iterrows():
            TTE = max(1e-6, 1 - (365-8+3+DAY+r.timestamp/1e6)/365)
            sqrtT = math.sqrt(TTE)
            m = math.log(K/r.S) / sqrtT
            iv = smile[0]*m**2 + smile[1]*m + smile[2]
            theo = bs_call(r.S, K, TTE, iv)
            theos.append(r.mkt - theo)
            spreads.append(r.ask_price_1 - r.bid_price_1 if pd.notna(r.ask_price_1) and pd.notna(r.bid_price_1) else np.nan)

        td = pd.Series(theos)
        mean_td = td.ewm(span=20, adjust=False).mean()
        signal = td - mean_td
        sp = pd.Series(spreads).dropna()
        print(f"  {sym}: theo_diff mean={td.mean():.2f}  std={td.std():.2f}  signal_std={signal.std():.2f}  spread_mean={sp.mean():.2f}")
        print(f"         |signal|>0.1: {(signal.abs()>0.1).mean():.0%}  >1.0: {(signal.abs()>1.0).mean():.0%}  >3.0: {(signal.abs()>3.0).mean():.0%}  max={signal.abs().max():.2f}")

print("\nDone.")
