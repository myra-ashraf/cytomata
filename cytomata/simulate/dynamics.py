import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Simple(object):
    """Simple model where blue light (input) directly leads to observable
    mCherry reporter (output) which also degrades over time."""
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
        return [self.mc]


class OptoLexA(object):
    """A two step inducible gene expression model.
    Components:
        1. Blue Light
        2. LexA-CIB1
        3. CRY2-VP16
        4. DBD-mCherry
    Process:
        1. LexA-CIB1 and CRY2-VP16 are constitutively expressed
        2. LexA binds to its DNA binding domain (DBD)
        3. Blue light induction dimerizes CIB1-CRY2
        4. VP16 promotes expression of mCherry reporter
    """
    def __init__(self, bl=0, cib=0, cry=0, c2c1=0, mc=0, p0=1.0, k0=0.01, k1=0.02, kb=0.001, d=0.05):
        """
        bl = blue light : manipulated variable
        cib = LexA-CIB1
        cry = CRY2-VP16
        mc = mCherry : process variable
        """
        self.bl = bl
        self.cib = cib
        self.cry = cry
        self.c2c1 = c2c1
        self.mc = mc
        self.p0 = p0
        self.k0 = k0
        self.k1 = k1
        self.kb = kb
        self.d = d

    def step(self):
        """Transition to next state based on current state of the system"""
        self.cib += self.p0 - self.d * self.cib + self.kb * self.c2c1
        self.cry += self.p0 - self.d * self.cry + self.kb * self.c2c1
        self.c2c1 += self.k0 * self.bl * self.cry * self.cib - self.d * self.c2c1 - self.kb * self.c2c1
        self.mc += self.p0 * self.k1 * self.c2c1 - self.d * self.mc
        return [self.mc, self.cib, self.cry, self.c2c1]


class OptoT7RNAP(object):
    pass


class OptoCreLoxP(object):
    pass


def main():
    """Example"""
    x = []
    y = []
    q = []
    r = []
    s = []

    timerange = np.arange(0, 200, 1)
    model = OptoLexA()
    for t in timerange:
        if t == 10:
            model.bl = 1.0
        xi = model.bl
        yi, qi, ri, si = model.step()
        x.append(xi)
        y.append(yi)
        q.append(qi)
        r.append(ri)
        s.append(si)

    plt.plot(timerange, x, label='light')
    plt.plot(timerange, y, label='mCherry')
    plt.plot(timerange, q, label='CIB1')
    plt.plot(timerange, r, label='CRY2')
    plt.plot(timerange, s, label='CIB1-CRY2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
