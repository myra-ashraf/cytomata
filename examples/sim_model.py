import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import matplotlib.pyplot as plt

from cytomata.process import FOPDT


fopdt = FOPDT()
x = []
y = []
obs = fopdt.reset(y0=0.1, K=2.26624925, tau=15.9868200, theta=3.20177855, max_iter=1000)
for i in range(100):
    action = np.random.choice([0.0, 1.0], replace=False, p=[0.5, 0.5])
    obs, rew, done, info = fopdt.step(action)
    x.append(i)
    y.append(obs)
    plt.clf()
    plt.plot(x, y)
    plt.title('Current Action: ' + str(action))
    plt.ylabel('Output')
    plt.xlabel('Step')
    plt.pause(5e-1)
