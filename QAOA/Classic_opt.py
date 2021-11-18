#########################################################################
# QPack                                                                 #
# Koen Mesman, TU Delft, 2021                                           #
# This file defines how classical optimizers are implemented for QAOA   #
# List of optimizers included:                                          #
# BFGS                                                                  #
# Random BFGS                                                           #
# BOBYQA                                                                #
# Direct_l                                                              #
# COBYLA                                                                #
# Random COBYLA                                                         #
# SHGO                                                                  #
# Nelder Mead                                                           #
# semi-random Nelder Mead                                               #
# Hybrid SHGO + Nelder Mead                                             #
# BFGS                                                                  #
# G MLSL                                                                #
# G MLSL LDS                                                            #
# ISRES                                                                 #
# NEWUOA                                                                #
# Simulated Annealing                                                   #
#########################################################################
from math import pi
import scipy.optimize as opt
import scipy.optimize
import nlopt
import numpy as np
from qaoa_def import *

# TODO: include backends as parameter


def max_cut_inv(params, graph, p):
    out = max_cut_circ(params, graph, p)
    if type(out) == list:
        return -out[0]
    else:
        return -out


def max_cut_norm(params, v, e, p, backend):
    graph = [v, e]  # graph for some reason doesn't pass normally
    out = max_cut_circ(params, graph, p, backend)
    return out


def max_cut_alt(params, b, v, e, p):  # alternative form to use nlopt, nl_opt sends second empty param for no reason
    graph = [v, e]
    out = max_cut_circ(params, graph, p)
    return out


def dsp_inv(params, graph, p):
    [v, e] = graph
    out = dsp_cost(params, v, e, p)
    return -out


def tsp(params, graph, p):  # redundant for now
    return cost_tsp(params, graph, p, 10000)


def rand_bfgs(init_param, graph, p, q_func):
    min_func = {
        'max-cut': max_cut_inv,
        'TSP': tsp
    }.get(q_func)

    v, edge_list = graph
    best = [0]
    for i in range(2 * v):
        param = np.array(np.random.uniform(low=0.0, high=2 * pi, size=2))
        tmp = bfgs(param, graph, p, q_func)
        if tmp[0] > best[0]:
            best = tmp
    return best


def bobyqa(init_param, graph, p, q_func):
    v, e = graph
    min_func = {
        'max-cut': max_cut_alt,
        'TSP': tsp
    }.get(q_func)
    print(init_param)
    opt = nlopt.opt(nlopt.LN_BOBYQA, 2)
    opt.set_max_objective(lambda a, b: min_func(a, b, v, e, p))
    opt.set_lower_bounds([0]*(2*p))
    opt.set_upper_bounds([2 * pi]*(2*p))
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5]*(2*p)))
    opt.set_ftol_rel(1e-5)
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    opt_val = opt.last_optimum_value()  # returns number of expected cuts
    return [opt_val, xopt]


def direct_l(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.GN_DIRECT_L, 2)
    opt.set_max_objective(lambda a, b: max_cut_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2 * pi, 2 * pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-4, 1e-4]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimum_value(), xopt]


def cobyla(init_param, graph, p, q_func):
    min_func = {
        'max-cut': max_cut_alt,
        'TSP': tsp
    }.get(q_func)

    v, e = graph
    opt = nlopt.opt(nlopt.LN_COBYLA, 2)
    opt.set_max_objective(lambda a, b: min_func(a, b, v, e, p))
    opt.set_lower_bounds([0]*(2*p))
    opt.set_upper_bounds([2 * pi]*(2*p))
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5]*(2*p)))
    opt.set_ftol_rel(1e-5)
    arr_param = np.array(init_param)
    print(arr_param)
    xopt = opt.optimize(arr_param)
    opt_val = opt.last_optimum_value()  # returns number of expected cuts

    return [opt_val, xopt]


def r_cobyla(init_param, graph, p, q_func):
    v, edge_list = graph
    best = [0, [0]]
    for i in range(2 * v):
        param = np.array(np.random.uniform(low=0.0, high=2 * pi, size=2))
        tmp = cobyla(param, graph, p, q_func)
        if tmp[0] > best[0]:
            best = tmp
    return best


def shgo_fun(init_param, graph, p, q_func, backend):  # simplicial homology global optimization
    bounds = [(0, pi), (0, 2 * pi)]
    res = []
    if q_func == 'mcp':
        min_func = max_cut_norm
        v, e = graph
        res = opt.shgo(min_func, bounds, args=(v, e, p, backend),
                       options={'ftol': 1e-10})  # perhaps v, e can be replaced with graph

    if q_func == 'dsp':
        min_func = dsp_cost
        v, e = graph
        res = opt.shgo(min_func, bounds, args=(v, e, p, backend),
                       options={'ftol': 1e-10})  # perhaps v, e can be replaced with graph
    if q_func == 'tsp':
        min_func = tsp
        v, A, D = graph
        res = opt.shgo(min_func, bounds, args=(graph, p, backend), options={'ftol': 1e-10})
    return [res.fun, res.x, res.nfev, res.nit]


def shgo_local(init_param, graph, p, q_func):  # use this variant for a list of local optima
    # bounds = [(0, 2 * pi), (0, 2 * pi)]
    bounds = [(0, pi), (0, 2 * pi)]
    res = []
    res = []
    if q_func == 'mcp':
        min_func = max_cut_norm
        v, e = graph
        res = opt.shgo(min_func, bounds, args=(v, e, p),
                       options={'ftol': 1e-8})  # perhaps v, e can be replaced with graph

    if q_func == 'dsp':
        min_func = dsp_cost
        v, e = graph
        res = opt.shgo(min_func, bounds, args=(v, e, p),
                       options={'ftol': 1e-8})  # perhaps v, e can be replaced with graph
    if q_func == 'tsp':
        min_func = tsp
        v, A, D = graph
        res = opt.shgo(min_func, bounds, args=(graph, p), options={'ftol': 1e-8})
    return [res.funl, res.xl]


def nm(init_param, graph, p, q_func):
    min_func = {
        'mcp': max_cut_inv,
        'dsp': dsp_inv,
        'tsp': tsp
    }.get(q_func)

    res = opt.minimize(min_func, init_param, args=(graph, p), method='nelder-mead',
                       options={'ftol': 1e-4, 'maxfev': 100, 'disp': False})
    print(res)
    return [-res.fun, res.x]


def semi_r_nm(init_param, graph, p, q_func):
    max_out = 0
    out = []
    for beta in np.arange(0, 1, 0.1):
        for gamma in np.arange(0, 2, 0.5):
            res = nm([beta, gamma], graph, p, q_func)
            if res[0] > max_out:
                max_out = res[0]
                out = res
    return out


def h_shgo_nm(init_param, graph, p, q_func):
    [lout, lpar] = shgo_local(init_param, graph, p, q_func)
    out_opt = 0
    par_opt = [0, 0]
    for par in lpar:
        res = nm(par, graph, p, q_func)
        if res[0] > out_opt:
            out_opt = res[0]
            par_opt = res[1]
    return [out_opt, par_opt]


def bfgs(init_param, graph, p, q_func):
    min_func = {
        'max-cut': max_cut_inv,
        'TSP': tsp
    }.get(q_func)
    res = opt.minimize(min_func, init_param, args=(graph, p), method='BFGS', options={'disp': False})

    return [1 / res.fun, res.x]


def g_mlsl(init_param, graph, p):
    v, e = graph
    opt_ml = nlopt.opt(nlopt.GD_MLSL, 2)
    opt_ml.set_max_objective(lambda a, b: max_cut_alt(a, b, v, e, p))
    opt_ml.set_lower_bounds([0, 0])
    opt_ml.set_upper_bounds([2 * pi, 2 * pi])
    opt_ml.set_initial_step(0.005)
    opt_ml.set_local_optimizer(opt.shgo)
    opt_ml.set_xtol_abs(np.array([1e-2, 1e-2]))
    arr_param = np.array(init_param)
    xopt = opt_ml.optimize(arr_param)
    return [opt_ml.last_optimize_result(), xopt]


def g_mlsl_lds(init_param, graph, p, q_app):
    v, e = graph

    opt_nm = nlopt.opt(nlopt.LN_NELDERMEAD, 2*p)
    opt_nm.set_ftol_rel(1e-5)
    opt_ml = nlopt.opt(nlopt.GD_MLSL_LDS, 2*p)
    opt_ml.set_max_objective(lambda a, b: max_cut_alt(a, b, v, e, p))
    opt_ml.set_lower_bounds([0]*(2*p))
    opt_ml.set_upper_bounds([2 * pi]*(2*p))
    opt_ml.set_initial_step(0.005)
    opt_ml.set_xtol_abs(np.array([1e-3]*(2*p)))
    opt_ml.set_local_optimizer(opt_nm)
    opt_ml.set_ftol_rel(1e-2)
    arr_param = np.array(init_param)
    xopt = opt_ml.optimize(arr_param)
    return [opt_ml.last_optimize_result(), xopt]


def isres(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.GN_ISRES, 2)
    opt.set_max_objective(lambda a, b: max_cut_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2 * pi, 2 * pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-3, 1e-3]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimize_result(), xopt]


def newuoa(init_param, graph, p):
    v, e = graph
    opt = nlopt.opt(nlopt.LN_NEWUOA_BOUND, 2)
    opt.set_max_objective(lambda a, b: max_cut_alt(a, b, v, e, p))
    opt.set_lower_bounds([0, 0])
    opt.set_upper_bounds([2 * pi, 2 * pi])
    opt.set_initial_step(0.005)
    opt.set_xtol_abs(np.array([1e-5, 1e-5]))
    arr_param = np.array(init_param)
    xopt = opt.optimize(arr_param)
    return [opt.last_optimize_result(), xopt]


def d_annealing(init_param, graph, p):
    bounds = [(0, 2 * pi), (0, 2 * pi)]
    v, e = graph
    res = scipy.optimize.dual_annealing(max_cut_alt, bounds, args=(graph, p))
    return [res.fun, res.x]
