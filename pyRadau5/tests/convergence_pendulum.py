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
tf = 3.5 * 0.33     # final time (one oscillation is ~3.5s long)
# rtol=1e-3; atol=rtol # relative and absolute tolerances for time adaptation
bPrint=False # if True, additional printouts from Radau during the computation
bDebug=False # study condition number of the iteration matrix


## Physical parameters for the pendulum
theta_0=0#np.pi/2 # initial angle
theta_dot0=-3. # initial angular velocity
r0=3.  # rod length
m=1.   # mass
g=9.81 # gravitational acceleration

dae_fun, jac_dae, mass, Xini, var_index = generateSystem(chosen_index=chosen_index, theta_0=theta_0, theta_dot0=theta_dot0, r0=r0, m=m, g=g)
n = Xini.size

var_index=np.array([0,0,2,2,3])

rtol=1e0
baseParameters={
  "max_newton_ite":50,
  "max_bad_ite":50,
  "bAlwaysApply2ndEstimate":True,
  "jacobianRecomputeFactor":-1,
  "bUseExtrapolatedGuess":True,
  "bUsePredictiveController":False,
  "safety_factor":0.9,
  "newton_tol": 1e-12/rtol,
  "min_factor": 0.2,
  "max_factor": 8.0,
  "rtol":rtol,
  "atol":rtol,
  # Scipy Radau specific
  "scale_residuals": True,
  "scale_newton_norm": True,
  "scale_error": True,
  "bUsePredictiveNewtonStoppingCriterion": False,
  "bPerformAllNewtonIterations": False,
  "zero_algebraic_error": True,
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
  "bPrintProgress":True,
  "var_index": var_index,
  "bDebug": False,
  "nmax_step": 100000,
  "constant_dt": True,
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

sols=[]
nt = np.unique(np.logspace(1,3,30).astype(int))
# nt = 2**np.array(range(3,int(np.log(1e4)/np.log(2))))
dt_vec = tf/(nt+1)

for dt in dt_vec:
  baseParameters['dt'] = dt
  baseParameters['first_step'] = dt
  baseParameters['max_step'] = dt
  sols.append( oneRun(baseParameters) )

#%% Compute reference solution
def fun_ode(t,X):
  theta=X[0]; theta_dot = X[1]
  return np.array([theta_dot,
                   -g/r0*np.sin(theta)])

Xini_ode= np.array([theta_0,theta_dot0])
sol_ode = solve_ivp(fun=fun_ode, t_span=(0., tf), y0=Xini_ode,
                rtol=1e-14, atol=1e-14, max_step=tf/1000, method='DOP853',
                dense_output=True)
theta_ode = sol_ode.y[0,:]
theta_dot = sol_ode.y[1,:]
x_ode =  r0*np.sin(theta_ode)
y_ode = -r0*np.cos(theta_ode)
vx_ode =  r0*theta_dot*np.cos(theta_ode)
vy_ode =  r0*theta_dot*np.sin(theta_ode)
T_ode = m*r0*theta_dot**2 + m*g*np.cos(theta_ode)
lbda_ode = T_ode / np.sqrt(x_ode**2+y_ode**2)

dicref = {'t':sol_ode.t,
           'x':x_ode,
           'y':y_ode,
           'vx':vx_ode,
           'vy':vy_ode,
           'theta':theta_ode,
           'lbda': lbda_ode,
           'T': T_ode}

sol_ode.y = np.vstack((x_ode, y_ode, vx_ode, vy_ode, lbda_ode))
sol_ode.sol_old = sol_ode.sol

def interpRef(t):
  ode_vars = sol_ode.sol_old(t)
  theta_ode = ode_vars[0,:]
  theta_dot = ode_vars[1,:]
  x_ode =  r0*np.sin(theta_ode)
  y_ode = -r0*np.cos(theta_ode)
  vx_ode =  r0*theta_dot*np.cos(theta_ode)
  vy_ode =  r0*theta_dot*np.sin(theta_ode)
  T_ode = m*r0*theta_dot**2 + m*g*np.cos(theta_ode)
  lbda_ode = T_ode / np.sqrt(x_ode**2+y_ode**2)
  return np.vstack((x_ode, y_ode, vx_ode, vy_ode, lbda_ode))

sol_ode.sol = interpRef
assert sol_ode.success

#%% Compute errors
# sol_ref = sols[-1]
import scipy.integrate
sol_ref = sol_ode
errors = np.zeros((Xini.size, nt.size))
errors_st = np.zeros_like(errors)
p=2 # which norm
for i, sol in enumerate(sols):
  if not sol.success:
    errors[:,i] = np.nan
    continue
  for ivar in range(Xini.size):
      errors[ivar,i] = abs((sol.y[ivar,-1] - sol_ref.y[ivar,-1]) / sol_ref.y[ivar,-1])
      interped_sol = sol_ref.sol(sol.t)
      rel_error = ( (sol.y[ivar,:] - interped_sol[ivar,:]) / (1e-10 + abs(interped_sol[ivar,:])) )
      errors_st[ivar,i] = (1/tf) * scipy.integrate.trapezoid(abs(rel_error)**p, sol.t)**(1/p)
      # errors_st[ivar,i] = (1/tf) * np.sum(abs(rel_error[1:])**p * np.diff(sol.t) )**(1/p)

#%% Compute actual dt
plt.figure()
for i, sol in enumerate(sols):
  plt.semilogy(sol.t[:-1], np.diff(sol.t))
plt.xlabel('t')
plt.ylabel('dt')
plt.grid()

#%% Plots
plt.figure()
for ivar in range(Xini.size):
  plt.loglog(dt_vec, errors[ivar,:], label=f'var {ivar}', marker='.')
  plt.xlabel('dt')
  plt.ylabel('error')
  plt.grid()
  plt.legend()
  plt.title('Error at last time point')
  
plt.figure()
for ivar in range(Xini.size):
  plt.semilogx(dt_vec, np.gradient(np.log(errors[ivar,:]), np.log(dt_vec)), label=f'var {ivar}', marker='.')
  plt.xlabel('dt')
  plt.ylabel('error')
  plt.grid()
  plt.legend()
  plt.ylim(0,7)
  plt.title('Convergence order at last time point')

#%% Plots
plt.figure()
for ivar in range(Xini.size):
  plt.loglog(dt_vec, errors_st[ivar,:], label=f'var {ivar}', marker='.')
  plt.xlabel('dt')
  plt.ylabel('error')
  plt.grid()
  plt.legend()
  plt.title(f'Global errors (L{p}-norm)')

  
plt.figure()
for ivar in range(Xini.size):
  plt.semilogx(dt_vec, np.gradient(np.log(errors_st[ivar,:]), np.log(dt_vec)), label=f'var {ivar}', marker='.')
  plt.xlabel('dt')
  plt.ylabel('error')
  plt.grid()
  plt.legend()
  plt.ylim(0,7)
  plt.title('Order of convergence, global error')
