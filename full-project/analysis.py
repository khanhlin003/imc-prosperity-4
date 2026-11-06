import pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

prices = pd.concat([pd.read_csv(f'/Users/linhtrankhanh/Downloads/imc competition/data/round4/prices_round_4_day_{d}.csv', sep=';').assign(day=d) for d in [1,2,3]])
trades = pd.concat([pd.read_csv(f'/Users/linhtrankhanh/Downloads/imc competition/data/round4/trades_round_4_day_{d}.csv', sep=';').assign(day=d) for d in [1,2,3]])

# VE price per day
for d in [1,2,3]:
    ve = prices[(prices['product']=='VELVETFRUIT_EXTRACT') & (prices['day']==d)]['mid_price']
    print(f"VE day {d}: start={ve.iloc[0]:.0f} end={ve.iloc[-1]:.0f} min={ve.min():.0f} max={ve.max():.0f} drift={ve.iloc[-1]-ve.iloc[0]:.0f} std={ve.std():.2f}")

print()
# HPACK price per day
for d in [1,2,3]:
    h = prices[(prices['product']=='HYDROGEL_PACK') & (prices['day']==d)]['mid_price']
    print(f"HPACK day {d}: start={h.iloc[0]:.0f} end={h.iloc[-1]:.0f} min={h.min():.0f} max={h.max():.0f} drift={h.iloc[-1]-h.iloc[0]:.0f} std={h.std():.2f}")

print()
# VE autocorrelation
ve_all = prices[prices['product']=='VELVETFRUIT_EXTRACT']['mid_price']
rets = ve_all.diff().dropna()
print(f"VE returns: mean={rets.mean():.4f} std={rets.std():.4f} skew={rets.skew():.3f}")
print(f"VE lag-1 autocorr: {rets.autocorr(1):.4f}")
print(f"VE lag-2 autocorr: {rets.autocorr(2):.4f}")
print(f"VE abs(ret)>2 fraction: {(rets.abs()>2).mean():.3f}")
print(f"VE abs(ret)>5 fraction: {(rets.abs()>5).mean():.3f}")
print()

# HPACK autocorr
hp_all = prices[prices['product']=='HYDROGEL_PACK']['mid_price']
hrets = hp_all.diff().dropna()
print(f"HPACK returns: mean={hrets.mean():.4f} std={hrets.std():.4f}")
print(f"HPACK lag-1 autocorr: {hrets.autocorr(1):.4f}")
print(f"HPACK spread (ask1-bid1) stats:")
hp_spread = prices[prices['product']=='HYDROGEL_PACK'].eval('ask_price_1 - bid_price_1')
print(hp_spread.describe())
