import numpy as np
import matplotlib.pyplot as plt


class Simple(object):
    """Simple first order model where input produces output at a linear rate
    and the output also degrades at the linear rate."""
    def __init__(self, A=0, B=0, k=0.5, d=0.05):
        """
        A = input
        B = output
        k = production rate of B from A
        d = degradation rate of B
        """
        self.A = A
        self.B = B
        self.k = k
        self.d = d

    def step(self):
        """Transition to next state."""
        self.B += self.k * self.A - self.d * self.B
        return [self.B]


class OptoLexA(object):
    """An inducible gene expression model based on cytoplasmic-nuclear
    localization of a transcription activator.
    Components:
        1. Blue Light
        2. LexA-VP64 (cytoplasm)
        3. LexA-VP64 (nucleus)
        4. LBS-mNeonGreen
    Process:
        1. LexA-VP64 is constitutively expressed; mostly localizes to cytoplasm
        2. Blue light induction increases LexA-VP64 localization to nucleus
        3. LexA binds to LexA Binding Site (LBS)
        4. VP64 increases expression of mNeonGreen reporter
    """
    def __init__(self, Lc=0, Ln=0, kL=0, kn=0, kc=0, kd=0, Gr=0, kG=0):
        """
        Lc = conc of LexA-VP64 in cytoplasm
        Ln = conc of LexA-VP64 in nucleus
        Gr = conc of mNeonGreen reporter
        kL = rate of LexA-VP64 (cytoplasm) production
        kn = rate of LexA-VP64 transport to nucleus from cytoplasm
        kc = rate of LexA-VP64 transport to cytoplasm from nucleus
        kd = rate of dilution due to cell growth and protein degradation
        """
        # Sensing Model
        self.Lc = Lc
        self.Ln = Ln
        self.kL = kL
        self.kn = kn
        self.kc = kc
        self.kd = kd
        # Output Model
        self.Gr = Gr
        self.kG = kG
        self.K = 0
        self.a = 0
        self.b = 0
        self.n = 0

    def step(self):
        """Transition to next state."""
        self.Lc += self.kL + self.kc * self.Ln - (self.kn + self.kd) * self.Lc
        self.Ln += self.kn * self.Lc - (self.kc + self.d) * self.Ln
        self.Gr += self.kG - self.kd * self.Gr
        R = self.Ln / self.Lc
        self.kG = self.b + self.a * (R**self.n / (self.K**self.n + R**self.n))
        return [self.Lc, self.Ln, self.Gr]


class OptoT7RNAP(object):
    pass


if __name__ == '__main__':
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
