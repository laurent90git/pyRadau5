#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 11:37:15 2022

Test the code's performance with various parametrisation

@author: laurent
"""

from scipy.integrate import solve_ivp
from numpy.testing import (assert_, assert_allclose, assert_equal, assert_no_warnings, suppress_warnings)
import matplotlib.pyplot as plt
import numpy as np
from scipyDAE.tests.recursive_pendulum import generateSystem
from pyRadau5 import integration
import time as pytime

def computeAngle(x,y):
  theta = np.arctan(-x/y)
  I = np.where(y>0)[0]
  theta[I] += np.sign(x[I])*np.pi
  return theta

# Test the pendulum
from scipy.integrate import solve_ivp
from numpy.testing import (assert_, assert_allclose,
                       assert_equal, assert_no_warnings, suppress_warnings)
import matplotlib.pyplot as plt

from scipyDAE.radauDAE import RadauDAE
# from radauDAE_subjac import RadauDAE

###### Parameters to play with
n = 50
initial_angle = np.pi/4
chosen_index = 3 # The index of the DAE formulation
rtol=1e-3; atol=rtol # relative and absolute tolerances for time adaptation
bPrint=False # if True, additional printouts from Radau during the computation

dae_fun, jac_dae, sparsity, mass, Xini, var_index, T_th, m = \
    generateSystem(n, initial_angle, chosen_index)

tf = 2*T_th # simulate 2 periods
jac_dae = None
bandwith = (10,9) #integration.bandwith(sparsity)

# analyse comaprative autour d'un point intriguant
# tf = 0.1
# Xini = np.array([-2.86803493, -0.87998615, 1.84133183, -6.00123533, 5.33755473])

#%% Numerical parameters
max_newton_ite = 7
bUseExtrapolatedGuess=True
bUsePredictiveController=True
safetyFactor=0.9
max_bad_ite=2
bAlwaysApply2ndEstimate=True
jacobianRecomputeFactor=0.001 #0.001
newton_tol=None # relative newton tolerance with respect to rtol
first_step=tf/2 #6
step_evo_factor_bounds=(0.2,8.0)
rtol=1e-8; atol=rtol # relative and absolute tolerances for time adaptation



# Fortran specific
deadzone=(0.999,1.0001)
bShowProgress=True
bReport=True

# Scipy Radau specific
bUsePredictiveNewtonStoppingCriterion = True
zero_algebraic_error = False
scale_residuals = True
scale_newton_norm = True
scale_error = True
factor_on_non_convergence = 0.5
bDebug=False

if 0:
    #%% Solve the DAE
    t_start = pytime.time()
    solfort = integration.radau5(tini=0., tend=tf, y0=Xini,
                        fun=dae_fun,
                        mljac=bandwith[0], mujac=bandwith[1],
                        mlmas=0, mumas=0,
                        rtol=rtol, atol=atol,
                        t_eval=None,
                        nmax_step = 10000,
                        max_step = tf,
                        first_step=min(tf, first_step),
                        max_ite_newton=max_newton_ite, bUseExtrapolatedGuess=bUseExtrapolatedGuess,
                        bUsePredictiveController=bUsePredictiveController, safetyFactor=safetyFactor,
                        deadzone=deadzone, step_evo_factor_bounds=step_evo_factor_bounds,
                        jacobianRecomputeFactor=jacobianRecomputeFactor, newton_tol=newton_tol,
                        mass_matrix=mass, var_index=var_index,
                        bPrint=bPrint, nMaxBadIte=max_bad_ite, bAlwaysApply2ndEstimate=bAlwaysApply2ndEstimate,
                        bReport=bReport, bShowProgress=True)
    t_end = pytime.time()
    sol=solfort
    if solfort.success:
        state='solved'
    else:
        state='failed'
    print("Fortran DAE of index {} {}".format(chosen_index, state))
    print("{} time steps ({} = {} accepted + {} rejected + {} failed)".format(
      sol.t.size-1, sol.nstep, sol.naccpt, sol.nrejct, sol.nfail))
    print("{} fev, {} jev, {} LUdec, {} linsolves, {} linsolves for error estimation".format(
          sol.nfev, sol.njev, sol.ndec, sol.nsol, sol.nlinsolves_err))
    print('CPU time = {} s'.format(t_end-t_start))

    if 0:
        # recover the time history of each variable
        # x,y,vx,vy,lbda = sol.y
        # T = lbda * np.sqrt(x**2+y**2)
        # theta = computeAngle(x,y)
        # dicfort = {'t':sol.t,
        #            'x':x,
        #            'y':y,
        #            'vx':vx,
        #            'vy':vy,
        #            'theta':theta,
        #            'lbda': lbda,
        #            'T': T}

        #%% Plot report
        if bReport:
            print('codes = ', np.unique(sol.reports['code']))

            codes, names =  zip(*
                            ((0, 'accepted'),
                             (1, 'rejected'),
                             (2, 'refactor_from_newton'),
                             (3, 'update_from_newton'),
                             (4, 'singular'),
                             (5, 'badconvergence'),
                             (-10, 'jacobian_update'),
                             (-11, 'refactor'),
                             ))
            Idict = {}
            for key, val in zip(names, codes):
              Idict[key] = np.where(sol.reports["code"]==val)[0]
            Idict['failed'] = np.where( np.logical_or(sol.reports["code"]==4, sol.reports["code"]==5))[0]

            plt.figure()
            plt.plot(sol.t[:-1], np.diff(sol.t), label=r'$\Delta t$')
            # plt.plot(sol.reports['t'][Idict['accepted']], sol.reports['dt'][Idict['accepted']], linestyle='--', label='dt2')

            # plt.plot(sol.reports['t'][Idict['refactor']],        sol.reports['dt'][Idict['refactor']], linestyle='', marker='o', color='tab:green', label='refactor', markersize=15, alpha=0.3)
            # plt.plot(sol.reports['t'][Idict['jacobian_update']], sol.reports['dt'][Idict['jacobian_update']], linestyle='', marker='o', color='tab:green', label='jacobian update', alpha=1)

            plt.plot(sol.reports['t'][Idict['jacobian_update']], sol.reports['dt'][Idict['jacobian_update']], linestyle='', marker='o', color='tab:green', label='jacobian update', markersize=15, alpha=0.3)
            plt.plot(sol.reports['t'][Idict['failed']],          sol.reports['dt'][Idict['failed']], linestyle='', marker='o', color='tab:red', label='failed')
            plt.plot(sol.reports['t'][Idict['rejected']],        sol.reports['dt'][Idict['rejected']], linestyle='', marker='o', color='tab:purple', label='rejected')
            plt.legend()
            plt.yscale('log')

            ax2 = plt.gca().twinx()
            ax2.plot(sol.reports['t'][Idict['accepted']], sol.reports['newton_iterations'][Idict['accepted']], label='all', color='tab:green', linestyle='--')
            ax2.plot(sol.reports['t'][Idict['accepted']], sol.reports['bad_iterations'][Idict['accepted']], label='bad', color='tab:red', linestyle='--')
            ax2.legend()
            ax2.set_ylabel('Newton iterations')
            plt.grid()
            plt.legend()
            plt.xlabel('t (s)')
            plt.ylabel('dt (s)')
            plt.title('Radau5 analysis')

else:
    #%% Solve the DAE with Scipy's modified Radau
    from scipyDAE.radauDAE import RadauDAE
    from scipyDAE.radauDAE import solve_ivp_custom as solve_ivp
    t_start = pytime.time()
    solpy = solve_ivp(fun=dae_fun, t_span=(0., tf), y0=Xini, max_step=tf,
                    # rtol=rtol, atol=atol, jac=None, jac_sparsity=sparsity,
                    rtol=rtol, atol=atol, jac=jac_dae, jac_sparsity=sparsity,
                    method=RadauDAE,
                    first_step=min(tf, first_step), dense_output=True,
                    mass_matrix=mass, bPrint=bPrint,
                    max_newton_ite=max_newton_ite, min_factor=step_evo_factor_bounds[0], max_factor=step_evo_factor_bounds[1],
                    factor_on_non_convergence = factor_on_non_convergence,
                    var_index=var_index,
                    newton_tol=newton_tol,
                    scale_residuals = scale_residuals,
                    scale_newton_norm = scale_newton_norm,
                    scale_error = scale_error,
                    zero_algebraic_error = zero_algebraic_error,
                    jacobianRecomputeFactor=jacobianRecomputeFactor,
                    bAlwaysApply2ndEstimate = bAlwaysApply2ndEstimate,
                    max_bad_ite=max_bad_ite,
                    bUsePredictiveNewtonStoppingCriterion=bUsePredictiveNewtonStoppingCriterion,
                    bDebug=bDebug, bPrintProgress=True)
    t_end = pytime.time()
    sol = solpy
    if sol.success:
        state='solved'
    else:
        state='failed'
    print("\nScipy DAE of index {} {}".format(chosen_index, state))
    print("{} time steps ({} = {} accepted + {} rejected + {} failed)".format(
      sol.t.size-1, sol.solver.nstep, sol.solver.naccpt, sol.solver.nrejct, sol.solver.nfailed))
    print("{} fev, {} jev, {} LUdec, {} linsolves, {} linsolves for error estimation".format(
          sol.nfev, sol.njev, sol.nlu, sol.solver.nlusove, sol.solver.nlusolve_errorest))
    print('CPU time = {} s'.format(t_end-t_start))

    x,y,vx,vy,lbda = sol.y
    T = lbda * np.sqrt(x**2+y**2)
    theta = computeAngle(x,y)
    dicpy   = {'t':sol.t,
               'x':x,
               'y':y,
               'vx':vx,
               'vy':vy,
               'theta':theta,
               'lbda': lbda,
               'T': T}

if 0:
    #%%
    plt.figure()
    plt.semilogy(solfort.t[:-1], np.diff(solfort.t), label='fortran')
    plt.semilogy(solpy.t[:-1], np.diff(solpy.t), label='py')
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('dt (s)')
    plt.legend()
    #%% Compute true solution (ODE on the angle in polar coordinates)
    def fun_ode(t,X):
      theta=X[0]; theta_dot = X[1]
      return np.array([theta_dot,
                       -g/r0*np.sin(theta)])

    Xini_ode= np.array([theta_0,theta_dot0])
    sol_ode = solve_ivp(fun=fun_ode, t_span=(0., tf), y0=Xini_ode,
                    rtol=1e-12, atol=1e-12, max_step=tf/10, method='DOP853',
                    dense_output=True)
    theta_ode = sol_ode.y[0,:]
    theta_dot = sol_ode.y[1,:]
    x_ode =  r0*np.sin(theta_ode)
    y_ode = -r0*np.cos(theta_ode)
    theta_ode2 = computeAngle(x_ode,y_ode)
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
               'theta2':theta_ode2,
               'lbda': lbda_ode,
               'T': T_ode}

    #%% Compare the DAE solution and the true solution

    plt.figure()
    plt.plot(dicref['t'],dicref['x'], label='x')
    plt.plot(dicref['t'],dicref['y'], label='y')
    plt.plot(dicref['t'],dicref['theta'], label='theta', linestyle='--')
    plt.grid()
    plt.legend()
    plt.title('True solution')


    #%% Compare the DAE solution and the true solution
    plt.figure()
    for sol,name,linestyle,marker,color in [
                           (dicfort,'fortran','-','+','tab:blue'),
                           (dicpy,'py',':','o','tab:orange'),
                            (dicref, 'ref', '--',None,'tab:green')
                           ]:
      plt.plot(sol['t'],sol['x'], color=color, linestyle=linestyle, marker=marker, label=None)#r'$x_{{}}$'.format(name))
      plt.plot(sol['t'],sol['y'], color=color, linestyle=linestyle, marker=marker, label=None)#r'$x_{{}}$'.format(name))
      # plt.plot(sol.t,y, color='tab:blue', linestyle='--', label=r'$y_{DAE}$')
      plt.plot(sol['t'],sol['y']**2+sol['x']**2, color=color, marker=marker, linestyle=linestyle, label=None)#r'$x_{{}}^2 + y_{{}}^2$'.format(name,name))
      plt.plot(np.nan, np.nan, linestyle=linestyle, color=color, marker=marker, label=name)
    plt.grid()
    plt.legend()
    plt.title('Comparison with the true solution')

    #%% Compare the DAE solution and the true solution
    plt.figure()
    for sol,name,linestyle,marker,color in [
                           (dicfort,'fortran','-','+','tab:blue'),
                           (dicpy,'py',':','o','tab:orange'),
                            (dicref, 'ref', '--',None,'tab:green')
                           ]:
      plt.plot(sol['t'],sol['T'], color=color, linestyle=linestyle, marker=marker, label=None)#r'$x_{{}}$'.format(name))
      plt.plot(np.nan, np.nan, linestyle=linestyle, color=color, marker=marker, label=name)
    plt.grid()
    plt.legend()
    plt.title('Joint force')

    #%% Analyze constraint violations
    # we check how well equations (5), (6) and (7) are respected
    fig,ax = plt.subplots(4,1,sharex=True)
    for sol,name,linestyle,marker,color in [
                       (dicfort,'fortran','-','+','tab:blue'),
                       (dicpy,'py',':','.','tab:orange'),
                        # (dicref, 'ref', '--',None,'tab:green')
                       ]:
      for i in range(3):
        t,x,y,vx,vy,lbda = sol['t'], sol['x'], sol['y'], sol['vx'], sol['vy'], sol['lbda']
        if i==0:
          constraint = lbda*(x**2+y**2)/m + g*y - (vx**2 + vy**2) #index 1
        if i==1:
          constraint = x*vx+y*vy #index 2
        if i==2:
          constraint = x**2 + y**2 - r0**2 #index 3

        # ax[i].plot(t, constraint, marker=marker, linestyle=linestyle, label=name, color=color)
        ax[i].semilogy(t, np.abs(constraint), marker=marker, linestyle=linestyle, label=name, color=color)

      ax[3].semilogy(t[:-1], np.diff(t), marker=marker, linestyle=linestyle, label=name, color=color)
    for i in range(4):
      ax[i].grid()
    for i in range(3):
      ax[i].set_ylabel('index {}'.format(i+1))
    ax[3].set_ylabel('dt')
    ax[-1].legend()
    ax[-1].set_xlabel('t (s)')
    fig.suptitle('Constraints violation')


    #%%
    plt.figure()
    for sol,name,linestyle,marker,color in [
                       (dicfort,'fortran','-','+','tab:blue'),
                       (dicpy,'py',':','.','tab:orange'),
                        (dicref, 'ref', '--',None,'tab:green')
                       ]:
      plt.plot(sol['t'], sol['theta'], marker=marker, linestyle=linestyle, label=name, color=color)
    plt.plot(dicref['t'], dicref['theta2'], marker=marker, linestyle=linestyle, label='ref2', color=[0,0,0])
    plt.plot(sol_ode.t, np.gradient(theta_ode,sol_ode.t), label='theta dot')
    plt.grid()
    plt.legend()
    plt.xlabel('t (s)')

    #%% Compute error
    for key in ['x','y','vx','vy', 'lbda']:
      plt.figure()
      plt.title(key)
      for sol,name,linestyle,marker,color in [
                       (dicfort,'fortran','-','+','tab:blue'),
                       (dicpy,'py',':','.','tab:orange'),
                       ]:
        ref_sol = sol_ode.sol(sol['t'])
        theta = ref_sol[0,:]
        theta_dot = ref_sol[1,:]
        refval ={"x": r0*np.sin(theta),
                 "y": -r0*np.cos(theta),
                 "vx": r0*theta_dot*np.cos(theta),
                 "vy": r0*theta_dot*np.sin(theta),
                 "T": m*r0*theta_dot**2 + m*g*np.cos(theta),
                 "theta": theta,
                 "theta_dot": theta_dot}
        refval["lbda"] = refval["T"]/ (refval["x"]**2 + refval["y"]**2)**0.5

        ref_value = refval[key]
        current_value = sol[key]

        rel_err = np.abs(ref_value - current_value) / (1e-5 + np.abs(ref_value))

        plt.semilogy(sol['t'], rel_err, label=name)
      plt.grid()
      plt.legend()
      plt.xlabel('t (s)')
