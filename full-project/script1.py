import pandas as pd
import numpy as np

df = pd.read_csv('data/round4/prices_round_4_day_1.csv', sep=';')
hpack = df[df['product']=='HYDROGEL_PACK'].copy().sort_values('timestamp')
hpack['obi'] = (hpack['bid_volume_1'] - hpack['ask_volume_1']) / (hpack['bid_volume_1'] + hpack['ask_volume_1'])
hpack['mid_ret'] = hpack['mid_price'].diff(1).shift(-1)  # next-tick return
# OBI quintile analysis
hpack['obi_q'] = pd.qcut(hpack['obi'], 5, labels=False, duplicates='drop')
print("Day 1 HPACK OBI quintile vs next-tick mid return:")
print(hpack.groupby('obi_q')['mid_ret'].agg(['mean','std','count']).round(3))
print()

# Day 2
df2 = pd.read_csv('data/round4/prices_round_4_day_2.csv', sep=';')
hpack2 = df2[df2['product']=='HYDROGEL_PACK'].copy().sort_values('timestamp')
hpack2['obi'] = (hpack2['bid_volume_1'] - hpack2['ask_volume_1']) / (hpack2['bid_volume_1'] + hpack2['ask_volume_1'])
hpack2['mid_ret'] = hpack2['mid_price'].diff(1).shift(-1)
hpack2['obi_q'] = pd.qcut(hpack2['obi'], 5, labels=False, duplicates='drop')
print("Day 2 HPACK OBI quintile vs next-tick mid return:")
print(hpack2.groupby('obi_q')['mid_ret'].agg(['mean','std','count']).round(3))
