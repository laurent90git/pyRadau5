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
from pyRadau5 import integration
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
# jac_dae = None
print(f'Solving the index {chosen_index} formulation')

# analyse comaprative autour d'un point intriguant
# tf = 0.1
# Xini = np.array([-2.86803493, -0.87998615, 1.84133183, -6.00123533, 5.33755473])

# #%% Numerical parameters
# max_newton_ite = 7
# max_bad_ite=2
# bAlwaysApply2ndEstimate=True
# jacobianRecomputeFactor=-1 #0.001 #0.001

# bUseExtrapolatedGuess=True
# bUsePredictiveController=True
# safetyFactor=0.9
# newton_tol=None # relative newton tolerance with respect to rtol
# first_step=tf/2 #6
# step_evo_factor_bounds=(0.2,8.0)
# rtol=1e-8; atol=rtol # relative and absolute tolerances for time adaptation

# # Scipy Radau specific
# scale_residuals = True
# scale_newton_norm = True
# scale_error = True

# bUsePredictiveNewtonStoppingCriterion = True
# zero_algebraic_error = False
# factor_on_non_convergence = 0.5
baseParameters={
  "max_newton_ite":7,
  "max_bad_ite":2,
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
  "bPrintProgress":True,
  "var_index": var_index,
  "bDebug": False,
  "nmax_step": 1000,
  }

parameter_sweeps = {
  "max_newton_ite":list(range(2,15)),#[3,5,7,10],
  "max_bad_ite":[0,1,2,3],
  "bAlwaysApply2ndEstimate":[False,True],
  "jacobianRecomputeFactor":[-1, 0.1],
  }

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
  # x,y,vx,vy,lbda = sol.y
  # T = lbda * np.sqrt(x**2+y**2)
  # theta = computeAngle(x,y)
  # dicpy   = {'t':sol.t,
  #            'x':x,
  #            'y':y,
  #            'vx':vx,
  #            'vy':vy,
  #            'theta':theta,
  #            'lbda': lbda,
  #            'T': T}

import itertools
import copy

swept_keys = list( parameter_sweeps.keys() )
swept_parameters = [parameter_sweeps[k] for k in parameter_sweeps.keys()]
params = copy.deepcopy( baseParameters )

sols = []
for current_set in itertools.product(*swept_parameters):
  # print(current_set)
  params=copy.deepcopy( baseParameters )
  for ik,k in enumerate(swept_keys):
    params[k] = current_set[ik]
  sols.append( oneRun( params ) )

#%% Rangement
# find a successful simulation
for i, sol, in enumerate(sols):
  if sol.success:
    break

allKeys = []
for k in sols[i].keys():
  if not isinstance(sols[i][k], np.ndarray):
    if sols[i][k] is not None:
      allKeys.append(k)
      
allKeys2 = list(sols[i].params.keys())
  
for k in ['message', 'solver', 'params']:
  allKeys.pop(allKeys.index(k))

data=dict((k,[]) for k in allKeys+allKeys2)
data['id']=[]

for isol, sol in enumerate(sols):
  data['id'].append(isol)
  for k in allKeys:
    data[k].append(sol[k])
  for k in allKeys2: # parameters
    data[k].append(sol.params[k])
    
# transform to Numpy arrays
for k in data.keys():
  data[k] = np.array(data[k])
    
#%%
choices =   [
             ("max_bad_ite", None),
             ("max_newton_ite", None),
             ("bAlwaysApply2ndEstimate", False),
             ("jacobianRecomputeFactor", 0.1),
            ]
chosen_keys, chosen_vals = zip(*choices)

# which keys are not fixe to a single value for visualisation
varying_keys = list(k for k,c in zip(chosen_keys, chosen_vals) if c is None)
assert sum((c is None for c in chosen_vals))==2, 'cannot use more than 2 varying parameters at once'

# find the subset of runs
Iselect = np.where( np.logical_and( *(data[chosen_keys[ic]]==c for ic,c in enumerate(chosen_vals) if c is not None)) )[0]

plt.figure()
reskey = 'CPUtime'
vark = varying_keys[0]
values = np.unique(data[vark][Iselect])
for val in values:
  Isub = np.where(data[vark][Iselect]==val)[0]
  I = Iselect[Isub]
  Ifail = np.where(data['success'][I]==False)[0]
  multiplier = np.ones((I.size))
  multiplier[Ifail] = np.nan
  plt.plot( data[varying_keys[1]][I], data[reskey][I]*multiplier, marker='.', label=f'{vark}={val}')

  # plt.plot( data[varying_keys[1]][Ifail], data[reskey][Ifail], marker='o', color='tab:red', label=None, linestyle='')
plt.legend()
plt.xlabel(varying_keys[1])
plt.ylabel(reskey)
plt.grid()

#%% Compute true solution (ODE on the angle in polar coordinates)
# def fun_ode(t,X):
#   theta=X[0]; theta_dot = X[1]
#   return np.array([theta_dot,
#                    -g/r0*np.sin(theta)])

# Xini_ode= np.array([theta_0,theta_dot0])
# sol_ode = solve_ivp(fun=fun_ode, t_span=(0., tf), y0=Xini_ode,
#                 rtol=1e-12, atol=1e-12, max_step=tf/10, method='DOP853',
#                 dense_output=True)
# theta_ode = sol_ode.y[0,:]
# theta_dot = sol_ode.y[1,:]
# x_ode =  r0*np.sin(theta_ode)
# y_ode = -r0*np.cos(theta_ode)
# theta_ode2 = computeAngle(x_ode,y_ode)
# vx_ode =  r0*theta_dot*np.cos(theta_ode)
# vy_ode =  r0*theta_dot*np.sin(theta_ode)
# T_ode = m*r0*theta_dot**2 + m*g*np.cos(theta_ode)
# lbda_ode = T_ode / np.sqrt(x_ode**2+y_ode**2)

# dicref = {'t':sol_ode.t,
#            'x':x_ode,
#            'y':y_ode,
#            'vx':vx_ode,
#            'vy':vy_ode,
#            'theta':theta_ode,
#            'theta2':theta_ode2,
#            'lbda': lbda_ode,
#            'T': T_ode}
