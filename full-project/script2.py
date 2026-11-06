import pandas as pd
import numpy as np
import math
from scipy.stats import norm

def bs_call(S, K, TTE, sigma):
    if sigma <= 0 or TTE <= 0: return 0.0, 0.0
    d1 = (math.log(S/K) + 0.5*sigma**2*TTE) / (sigma*TTE**0.5)
    d2 = d1 - sigma*TTE**0.5
    return S*norm.cdf(d1) - K*norm.cdf(d2), norm.cdf(d1)

def implied_vol(S, K, TTE, mkt_px):
    intrinsic = max(0.0, S-K)
    if mkt_px <= intrinsic+1e-6: return None
    lo, hi = 1e-4, 5.0
    for _ in range(50):
        mid = 0.5*(lo+hi)
        px, _ = bs_call(S, K, TTE, mid)
        if px > mkt_px: hi = mid
        else: lo = mid
        if hi-lo < 1e-5: break
    return 0.5*(lo+hi)

DAYS_PER_YEAR = 365
# The original code had a complex TTE formula. I'll stick to a simpler version if possible, 
# but I'll replicate the core logic provided.
# TTE = max(1e-6, 1 - (DAYS_PER_YEAR-8+DAY+ts//100/10_000)/DAYS_PER_YEAR) 
# Wait, DAY was defined as 4 in the script. Let's just use the user's logic exactly.

active = ['VEV_5100','VEV_5200','VEV_5400']
results = []
for day_n in [1,2,3]:
    try:
        df = pd.read_csv(f'data/round4/prices_round_4_day_{day_n}.csv', sep=';')
    except:
        continue
    ve = df[df['product']=='VELVETFRUIT_EXTRACT'].set_index('timestamp')['mid_price']
    for sym in active:
        sub_opt = df[df['product']==sym]
        if sub_opt.empty: continue
        opt = sub_opt.set_index('timestamp')['mid_price']
        for ts in opt.index:
            if ts not in ve.index: continue
            S = ve[ts]; K = int(sym.split('_')[-1])
            # Following the user's TTE logic:
            DAY_OFFSET = 4 # Provided in script
            TTE = max(1e-6, 1 - (DAYS_PER_YEAR-8+DAY_OFFSET+ts/100/10_000)/DAYS_PER_YEAR)
            iv = implied_vol(S, K, TTE, opt[ts])
            if iv: results.append({'day':day_n,'sym':sym,'ts':ts,'iv':iv,'S':S,'K':K,'TTE':TTE})

if results:
    res = pd.DataFrame(results)
    print("Mean IV by day and strike:")
    print(res.pivot_table('iv','day','sym',aggfunc='mean').round(4))
    print()
    print("IV std by day and strike:")
    print(res.pivot_table('iv','day','sym',aggfunc='std').round(4))
    print()
    print("IV trend within day (corr with timestamp):")
    for day_n in sorted(res['day'].unique()):
        for sym in sorted(res['sym'].unique()):
            sub = res[(res['day']==day_n)&(res['sym']==sym)]
            if len(sub) > 1:
                c = sub['ts'].corr(sub['iv'])
                print(f"  Day {day_n} {sym}: IV-ts corr = {c:.3f}")
else:
    print("No IV results calculated.")
