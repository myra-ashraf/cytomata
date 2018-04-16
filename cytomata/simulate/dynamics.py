import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Simple(object):
    """Simple model where blue light (input) directly yields observable
    mCherry protein (output)"""
    def __init__(self, bl=0, mc=0, k=0.5, d=0.05):
        """
        bl = blue light : manipulated variable
        mc = mCherry : process variable
        """
        self.bl = bl
        self.mc = mc
        self.k = k
        self.d = d

    def step(self):
        """Transition to next state based on current state of the system"""
        self.mc += self.k * self.bl - self.d * self.mc
        return self.mc


class OptoT7RNAP(object):
    pass


def main():
    """Example"""
    x = []
    y = []
    timerange = np.arange(0, 100, 1)
    model = SimpleModel()
    for t in timerange:
        if t == 10:
            model.bl = 1.0
        xi = model.bl
        yi = model.step()
        x.append(xi)
        y.append(yi)

    plt.plot(t, x)
    plt.plot(t, y)
    plt.show()


if __name__ == '__main__':
    main()
