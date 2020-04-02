from scikits.odes import ode
from scipy.integrate import solve_ivp


def sim_ind_translo(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        a = params['a']
        dy[0] = kr*y[1] - (u*ku + kf)*y[0]
        dy[1] = -dy[0]*a
    options = {'rtol': 1e-10, 'atol': 1e-15, 'max_steps': 50000}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_ind_dimer(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        dy[0] = -(u*ku + kf)*y[0]*y[1] + kr*y[2]
        dy[1] = -(u*ku + kf)*y[0]*y[1] + kr*y[2]
        dy[2] = (u*ku + kf)*y[0]*y[1] - kr*y[2]
    options = {'rtol': 1e-9, 'atol': 1e-12, 'max_steps': 50000}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_ind_gex(t, y0, Af, params):
    def model(t, y, dy):
        A = Af(t)
        ka = params['ka']
        kb = params['kb']
        kc = params['kc']
        n = params['n']
        dy[0] = (ka*A**n)/(kb**n + A**n) - kc*y[0]
    options = {'rtol': 1e-9, 'atol': 1e-12, 'max_steps': 50000}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y
