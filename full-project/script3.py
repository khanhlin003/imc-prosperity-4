import pandas as pd
import numpy as np

results_m01 = []
for day_n in [1,2,3]:
    trades = pd.read_csv(f'data/round4/trades_round_4_day_{day_n}.csv', sep=';')
    m01_otm = trades[(trades['buyer']=='Mark 01') & (trades['symbol'].isin(['VEV_5300','VEV_5400','VEV_5500','VEV_6000','VEV_6500']))]
    print(f"Day {day_n} Mark 01 OTM call buys: {len(m01_otm)} trades")
    if len(m01_otm) > 0:
        print(m01_otm[['timestamp','symbol','price','quantity']].to_string())
    print()
