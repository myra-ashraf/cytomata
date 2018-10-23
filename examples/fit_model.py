import os
import sys
sys.path.append(os.path.abspath('../'))

import lmfit as lm
import pandas as pd

from cytomata.process import FOPDTFitter


df = pd.read_csv('data.csv')
tp = df['time'].values/3600.
up = df['light'].values
yp = df['fluo'].values*10
fitter = FOPDTFitter(tp, up, yp)
fitter.optimize()
