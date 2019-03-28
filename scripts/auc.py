import numpy as np
import pandas as pd
from scipy.integrate import simps


df = pd.read_csv('data/yijia_data.csv')
t = df['Time']
ys = df.loc[:, df.columns != 'Time']
ratios = []
for col in ys:
    y = ys[col] - ys[col].min()
    auc = simps(y, t)
    if np.isnan(auc):
        auc = simps(y[~np.isnan(y)], t[~np.isnan(y)])
    rect = t.max() * y.max()
    ratios.append(auc/rect)
ratios = pd.Series(ratios).to_csv('data/yijia_data_results.csv')
