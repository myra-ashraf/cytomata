import numpy as np
import matplotlib.pyplot as plt

from model import FOPDT


def residual(on=1, off=1):
    fopdt = FOPDT()
    obs = fopdt.reset(y0=0.1, K=2.26624925, tau=15.9868200, theta=3.20177855, max_iter=1000)
    dt = 3.20177855/10.0
    dt_on = on * dt
    dt_off = off * dt
    reps = 360
    reps = (on + off) * int(np.round(reps/(on + off)))
    # print(reps)
    actions = ([1]*on + [0]*off)*int(np.round(reps/(on + off)))
    for i in range(reps):
        action = actions[i]
        # action = np.random.choice([0.0, 1.0], replace=False, p=[0.5, 0.5])
        fopdt.step(action)
    # plt.plot(fopdt.t, fopdt.y)
    # plt.show()
    # print(np.mean(np.ediff1d(fopdt.y)))
    # return (np.mean(np.ediff1d(fopdt.y)) - 0.002)**2
    return sum([1.0 for i in np.ediff1d(fopdt.y) if i < 0])


if __name__ == '__main__':
    # res = residual(10, 1)
    # print(res)
    ons = range(0, 11, 1)
    offs = range(1, 11, 1)
    best_res = 1e6
    best_on = None
    best_off = None
    for on in ons:
        for off in offs:
            res = residual(on, off)
            if res < best_res:
                best_res = res
                best_on = on
                best_off = off
    print(best_on, best_off, best_res)
