import os

import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
plt.rcParams['figure.figsize'] = 12, 12
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelpad'] = 0

from cytomata.utils.io import setup_dirs


def sim_ind_dimer(t, y0, u, params):
    def ind_dimer_model(t, y, uf, params):
        u = uf(t)
        [A, B, AB] = y
        kf = params['kf']
        kr = params['kr']
        dA = kr*AB - u*kf*A*B
        dB = kr*AB - u*kf*A*B
        dAB = u*kf*A*B - kr*AB
        return [dA, dB, dAB]
    uf = interp1d(t, u)
    result = solve_ivp(
        fun=lambda t, y: ind_dimer_model(t, y, uf, params),
        t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-8, atol=1e-8)
    A = result.y[0]
    B = result.y[1]
    AB = result.y[2]
    return A, B, AB


def sim_ind_translo(t, y0, u, params):
    def ind_translo_model(t, y, uf, params):
        u = uf(t)
        [C, N] = y
        kf = params['kf']
        kr = params['kr']
        dC = kr*N - u*kf*C
        dN = u*kf*C - kr*N
        return [dC, dN]
    uf = interp1d(t, u)
    result = solve_ivp(
        fun=lambda t, y: ind_translo_model(t, y, uf, params),
        t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-8, atol=1e-8)
    C = result.y[0]
    N = result.y[1]
    return C, N


def sim_ind_gex(t, y0, u, params):
    def ind_gex_model(t, y, uf, params):
        u = uf(t)
        [P] = y
        ka = params['ka']
        kb = params['kb']
        kc = params['kc']
        n = params['n']
        dP = (ka*T**n)/(kb**n + T**n) - kc*P
        return [dP]
    uf = interp1d(t, u)
    result = solve_ivp(
        fun=lambda t, y: ind_gex_model(t, y, uf, params),
        t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-8, atol=1e-8)
    P = result.y[0]
    return P


def sim_LINTAD(t, y0, u, params):
    def model_LINTAD(t, y, uf, params):
        u = uf(t)
        [LCBc, LCBn, NCV, AC, POI] = y
        k1f = params['k1f']
        k1r = params['k1r']
        k2f = params['k2f']
        k2r = params['k2r']
        ka = params['ka']
        kb = params['kb']
        kc = params['kc']
        n = params['n']
        dLCBc = k1r*LCBn - u*k1f*LCBc
        dLCBn = u*k1f*LCBc + k2r*AC - k1r*LCBn - u*k2f*LCBn*NCV
        dNCV = k2r*AC - u*k2f*LCBn*NCV
        dAC = u*k2f*LCBn*NCV - k2r*AC
        dPOI = (ka*AC**n)/(kb**n + AC**n) - kc*POI
        return [dLCBc, dLCBn, dNCV, dAC, dPOI]
    uf = interp1d(t, u)
    result = solve_ivp(
        fun=lambda t, y: model_LINTAD(t, y, uf, params),
        t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-8, atol=1e-8)
    print('Num Func Evals: ' + str(result.nfev))
    return result.y


def sim_I1FFL(t, y0, u, params):
    def model_I1FFL(t, y, uf, params):
        u = uf(t)
        [Xi, Xa, Y, Z] = y
        kXf = params['kXf']
        kXr = params['kXr']
        kAa = params['kAa']
        kAb = params['kAb']
        kIa = params['kIa']
        kIb = params['kIb']
        kc = params['kc']
        n = params['n']
        dXi = kXr*Xa - u*kXf*Xi
        dXa = u*kXf*Xi - kXr*Xa
        dY = (kAa*Xa**n)/(kAb**n + Xa**n) - kc*Y
        dZ = (kAa*Xa**n)/(kAb**n + Xa**n) * kIa/(1 + (Y/kIb)**n) - kc*Z
        return [dXi, dXa, dY, dZ]
    uf = interp1d(t, u)
    result = solve_ivp(
        fun=lambda t, y: model_I1FFL(t, y, uf, params),
        t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-8, atol=1e-8)
    print('Num Func Evals: ' + str(result.nfev))
    return result.y


def sim_FRC(t, y0, u, params):
    def model_FRC(t, y, uf, params):
        u = uf(t)
        [N, P1, P2, M1, M2, G1, I1, G2, I2] = y
        kM1f = params['kM1f']
        kM1r = params['kM1r']
        kM2f = params['kM2f']
        kM2r = params['kM2r']
        kGa = params['kGa']
        kGb = params['kGb']
        n1 = params['n1']
        kIa = params['kIa']
        kIb = params['kIb']
        n2 = params['n2']
        dN = kM1r*M1 + kM2r*M2 - u*kM1f*N*P1 - u*kM2f*N*P2
        dP1 = kM1r*M1 - u*kM1f*N*P1
        dP2 = kM2r*M2 - u*kM2f*N*P2
        dM1 = u*kM1f*N*P1 - kM1r*M1
        dM2 = u*kM2f*N*P2 - kM2r*M2
        dG1 = (kGa*M1**n1)/(kGb**n1 + M1**n1) * kIa/(1 + (I2/kIb)**n2)
        dI1 = dG1
        dG2 = (kGa*M2**n1)/(kGb**n1 + M2**n1) * kIa/(1 + (I1/kIb)**n2)
        dI2 = dG2
        return [dN, dP1, dP2, dM1, dM2, dG1, dI1, dG2, dI2]
    uf = interp1d(t, u)
    result = solve_ivp(
        fun=lambda t, y: model_FRC(t, y, uf, params),
        t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-8, atol=1e-8)
    return result.y