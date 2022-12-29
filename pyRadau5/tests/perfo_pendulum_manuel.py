#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 11:37:15 2022

Test the code's performance with various parametrisation

@author: laurent
"""
from numpy.testing import (assert_, assert_allclose, assert_equal, assert_no_warnings, suppress_warnings)
import matplotlib.pyplot as plt
import numpy as np
from scipyDAE.tests.pendulum import generateSystem, computeAngle
# from pyRadau5 import integration
import time as pytime
from scipyDAE.radauDAE import RadauDAE
from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp

###### Parameters to play with
chosen_index = 3 # The index of the DAE formulation
tf = 10*3.5        # final time (one oscillation is ~3.5s long)
# rtol=1e-3; atol=rtol # relative and absolute tolerances for time adaptation
bPrint=False # if True, additional printouts from Radau during the computation
bDebug=False # study condition number of the iteration matrix


## Physical parameters for the pendulum
theta_0=0#np.pi/2 # initial angle
theta_dot0=-3. # initial angular velocity
r0=3.  # rod length
m=1.   # mass
g=9.81 # gravitational acceleration

dae_fun, jac_dae, mass, Xini, var_index= generateSystem(chosen_index=chosen_index, theta_0=theta_0, theta_dot0=theta_dot0, r0=r0, m=m, g=g)
n = Xini.size

# var_index=np.array([0,0,2,2,3])

baseParameters={
  "max_newton_ite":8,
  "max_bad_ite":None,
  "bAlwaysApply2ndEstimate":True,
  "jacobianRecomputeFactor":-1,
  "bUseExtrapolatedGuess":True,
  "bUsePredictiveController":True,
  "safety_factor":0.9,
  "newton_tol":None,
  "min_factor": 0.2,
  "max_factor": 8.0,
  "rtol":1e-4,
  "atol":1e-4,
  # Scipy Radau specific
  "scale_residuals": True,
  "scale_newton_norm": True,
  "scale_error": True,
  "bUsePredictiveNewtonStoppingCriterion": True,
  "zero_algebraic_error": False,
  "factor_on_non_convergence": 0.5,
  "fun": dae_fun,
  "jac": None,
  "jac_sparsity": None,
  "t_span": (0.,tf),
  "y0": Xini,
  "dense_output": False,
  "method": RadauDAE,
  "max_step": tf/2,
  "first_step": tf/2,
  "mass_matrix":mass,
  "bPrint":False,
  "bPrintProgress":False,
  "var_index": var_index,
  "bDebug": False,
  "nmax_step": 10000,
  }

# parameter_sweeps = {
#   "max_newton_ite":list(range(2,15)),#[3,5,7,10],
#   "max_bad_ite":[0,1,2,3],
#   "bAlwaysApply2ndEstimate":[False,True],
#   "jacobianRecomputeFactor":[-1, 0.1],
#   }

#%% Solve the DAE with Scipy's modified Radau
def oneRun(params):
  t_start = pytime.time()
  sol = solve_ivp(**params)
  t_end = pytime.time()
  sol.CPUtime = t_end - t_start
  sol.params = params
  sol.nlusolve = sol.solver.nlusolve
  sol.nlusolve_errorest = sol.solver.nlusolve_errorest
  sol.nstep = sol.solver.nstep
  sol.naccpt = sol.solver.naccpt
  sol.nfailed = sol.solver.nfailed
  sol.nrejct = sol.solver.nrejct
  return sol

for max_bad_ite in [0,1,2,3]:
  baseParameters['max_bad_ite'] = max_bad_ite
  sol = oneRun(baseParameters)
  print("\n {} bad iteration\n\tDAE {} in {} s,\n\t{} steps ({} = {} accepted + {} rejected + {} failed),\n\t{} fev, {} jev,\n\t{} LUdec, {} linsolves (+{} for error estimation)".format(
            max_bad_ite,
            'solved'*sol.success+(1-sol.success)*'FAILED', sol.CPUtime,
            sol.t.size-1, sol.nstep, sol.naccpt, sol.nrejct, sol.nfailed,
            sol.nfev, sol.njev, sol.nlu, sol.nlusolve, sol.nlusolve_errorest,
            ))