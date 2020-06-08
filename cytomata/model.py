from scikits.odes import ode
from scikits.odes.odeint import odeint


def sim_itranslo(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        a = params['a']
        dy[0] = -(u*ku + kf)*y[0] + kr*y[1]
        dy[1] = -a*dy[0]
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_idimer(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        dy[0] = -(u*ku + kf)*y[0]*y[1] + kr*y[2]
        dy[1] = dy[0]
        dy[2] = -dy[0]
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size': 10}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_TF(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        k1u = params['k1u']
        k1f = params['k1f']
        k1r = params['k1r']
        a = params['a']
        k2u = params['k2u']
        k2f = params['k2f']
        k2r = params['k2r']
        dy[0] = -(u*k1u + k1f)*y[0] + k1r*y[1]
        dy[1] = -(u*k2u + k2f)*y[1]*y[2] + k2r*y[3]
        dy[2] = -a*dy[0] - dy[1]
        dy[3] = -dy[1]
    options = {'rtol': 1e-10, 'atol': 1e-15}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_iexpress(t, y0, xf, params):
    def model(t, y, dy):
        X = xf(t)
        ka = params['ka']
        kb = params['kb']
        kc = params['kc']
        n = params['n']
        kf = params['kf']
        kg = params['kg']
        dy[0] = (ka*X**n)/(kb**n + X**n) - kc*y[0]
        dy[1] = kf*y[0] - kg*y[1]
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_ssl(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        [Ai, Bi, C, Aa, Ba, CA, CB] = y
        kua = params['ku']
        kra = params['kra']
        kaa = params['kaa']
        kda = params['kd']
        kub = params['ku']
        krb = params['krb']
        kab = params['kab']
        kdb = params['kd']
        dy[0] = -u*kua*Ai + kra*Aa
        dy[1] = -u*kub*Bi + krb*Ba
        dy[2] = -kaa*Aa*C - kab*Ba*C + kda*CA + kdb*CB
        dy[3] = u*kua*Ai - kra*Aa + kda*CA - kaa*Aa*C
        dy[4] = u*kub*Bi - krb*Ba + kdb*CB - kab*Ba*C
        dy[5] = -kda*CA + kaa*Aa*C
        dy[6] = -kdb*CB + kab*Ba*C
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y