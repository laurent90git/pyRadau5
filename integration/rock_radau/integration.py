#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import ctypes as ct

# try:
#   from scipy.linalg import bandwith
# except ImportError:
def bandwith(matrix):
    lband=uband=0
    assert matrix.shape[0]==matrix.shape[1], "only square mass matrices are allowed !"
    n = matrix.shape[0]
    for i in range(n):
      for j in range(i):
        if matrix[i,j]!=0:
          lband = max(lband, i-j)
      for j in range(i,n):
        if matrix[i,j]!=0:
          uband = max(uband, j-i)
    if uband==n-1:
      uband=n
    if lband==n-1:
      lband=n
    return (lband, uband)
    

class radau_result:
    def __init__(self, y, t):
        self.y = y
        self.t = t



def radau5(tini, tend, y0, fun,
    mujac, mljac,
    mass_matrix=None, mlmas=None, mumas=None, var_index=None,
    rtol=1e-4, atol=1e-4, t_eval=None,
    nmax_step=np.iinfo(np.int32).max, max_step=None, first_step=0.,
    max_ite_newton=None, bUseExtrapolatedGuess=None, bUsePredictiveController=None,
    jacobianRecomputeFactor=None, newton_tol=None, bTryToUseHessenberg=False,
    deadzone=None, step_evo_factor_bounds=None, safetyFactor=None,
    bPrint=False, bDebug=False, nMaxBadIte=0, bAlwaysApply2ndEstimate=False,
    bReport=False):
    """ TODO    """

    import os
    current_dir = os.path.dirname(os.path.realpath(__file__))
    subpath = "mylib/lib_radau_rock.so"
    sPath = os.path.join(current_dir, subpath)
    c_integration = ct.CDLL(sPath)

    neq = y0.size # number of components
    tsol=[]
    ysol=[]


    def fcn(n, t, y, ydot, rpar, ipar):
        y_np = np.ctypeslib.as_array(y, shape=(n[0],)) # transform input into a Numpy array
        ydot_np = fun(t[0], y_np) # call Python time derivative function #TODO: optional arguments *args
        for i in range(n[0]): # transform back from Numpy #TODO: faster way ?
          ydot[i] = ydot_np[i]

    if mass_matrix is None:
      # no mass matrix supplied --> set to identity
      mlmas = mumas = 0
      imas = 0
    else:
      if mlmas is None:
        mlmas, mumas = bandwith( mass_matrix ) # determine bandwith
      imas = 1

    def mass_fcn(n, am_in, lmas, rpar, ipar):
        am = np.ctypeslib.as_array(am_in, shape=(lmas[0], n[0]))
        #am = np.reshape(am_mapped, (lmas[0],n[0]), order='F')
        #print('am reshaped=\n', am)
        
        if bDebug:
          print('Calling mass matrix function')
          print('lmas=',lmas[0])
          print('n=',n[0])

        assert n[0]==neq
        am[:] = 5555555.
        
        
        ## To study the transposition issues from Fortan to C
        #ii=0
        #for i in range(n[0]):
        #  for j in range(lmas[0]):
        #    ii+=1
        #    am[i][j] = ii
        #print('am=\n',am)
        #return
        
        if mlmas==neq and mumas==neq: # full matrix
          am[:] = mass_matrix[:]

        else: # banded matrix
          mbdiag = mumas+1
  
          if 0:
              for i in range(lmas[0]):
                for j in range(n[0]):
                  print('i=',i+j-mbdiag+1, 'j=',j)
                  am[i][j] = mass_matrix[i+j-mbdiag+1,j]
          
          else:
              for diag_num in range(-mlmas,mumas+1,1):
                  current_diag = np.diag(mass_matrix, k=diag_num)
                  #print('diag_offset=', diag_num)
                  #print('  diag.size=',current_diag.size)
                  if diag_num<0: # lower diagonals
                    am[mbdiag - 1 - diag_num][:diag_num] = current_diag
                  elif diag_num>0: # upper diagonals
                    am[mbdiag - 1 - diag_num][diag_num:] = current_diag
                  else: # main diagonal
                    am[mbdiag - 1 - diag_num][:] = current_diag
          if 0:
              # DEBUG
              print(mass_matrix)
              for i in range(-mlmas,mumas+1):
                print('diag',i)
                print(am[i+mlmas][:])

              for i in range(mlmas+mumas+1):
                print(f'AM({i+1},:) = ', am[i][:])
        
        # to correct an issue with the fact that is array is not ordered in the same manner as in fortran
        #if 0:
        #    vector = np.reshape(am, (n[0]*lmas[0]), order='F')
        #    vector = vector.reshape((n[0],lmas[0]), order='F').T          
        #    am[:] = vector[:]
        #else:
        am[:] = np.reshape(am, (n[0], lmas[0]), order='F').T
        return
                  
    def solout(nr, told, t, y, cont, lrc, n, rpar, ipar, irtrn):
        if bPrint: print(f'solout called after step tn={told[0]} to tnp1={t[0]}')
        tsol.append(t[0])
        y_np = np.ctypeslib.as_array(y, shape=(n[0],)).copy() # transform input into a Numpy array
        ysol.append(y_np)
        irtrn[0] = 1 # if <0, Radau5 will exit --> TODO: handle events with that ?
    if bReport:
        nReport=1
        reports={"t":[],"dt":[],"code":[],'newton_iterations':[], 'bad_iterations':[]}
    else:
        nReport=0
        reports=None
    def reportfun(t,dt,code,newt,nbad):
        # print('Reporting',t[0],dt[0],code[0])
        reports['t'].append(t[0])
        reports['dt'].append(dt[0])
        reports['code'].append(code[0])
        reports['newton_iterations'].append(newt[0])
        reports['bad_iterations'].append(nbad[0])


    fcn_type = ct.CFUNCTYPE(None, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int))



    mas_fcn_type = ct.CFUNCTYPE(None, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int),
                            ct.POINTER(ct.c_int), ct.POINTER(ct.c_int))
    solout_type = ct.CFUNCTYPE(None, ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                               ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int),
                               ct.POINTER(ct.c_int), ct.POINTER(ct.c_double), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int))
    report_fcn_type = ct.CFUNCTYPE(None, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
                                   ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int))

    c_radau5 = c_integration.radau5_integration
    c_radau5.argtypes = [ct.c_double, ct.c_double, ct.c_double,
                         ct.c_int,  np.ctypeslib.ndpointer(dtype = np.float64), np.ctypeslib.ndpointer(dtype = np.float64),
                         fcn_type, mas_fcn_type, solout_type, report_fcn_type, ct.c_int,
                         ct.c_double, ct.c_double,
                         ct.c_int, ct.c_int,
                         ct.c_int, ct.c_int, ct.c_int, np.ctypeslib.ndpointer(dtype = np.int32),
                         np.ctypeslib.ndpointer(dtype = np.int32), np.ctypeslib.ndpointer(dtype = np.float64),
                         ct.c_int, np.ctypeslib.ndpointer(dtype = np.int32),
                         ct.c_int, ct.c_int, ct.c_int]
    c_radau5.restype = None

    # create callable interfaces to the Python functions (time derivatives, jacobian, solution export, mass matrix)
    callable_fcn = fcn_type(fcn)
    callable_mass_fcn = mas_fcn_type(mass_fcn)
    callable_solout = solout_type(solout)
    callable_reportfun = report_fcn_type(reportfun)

    yn = np.zeros(neq)
    info= np.zeros(20, dtype=np.int32)

    itol=0 # tolerances are specified as scalars
    assert np.isscalar(rtol) and np.isscalar(atol), "`rtol` and `atol` must be scalars"

    jac = None # jacobian function
    ijac = 0
    mujac=neq
    mljac=neq
    bDenseJacobian = (mujac == neq) and (mljac == neq) # Jacobian is dense ?



    ### FILL IN PARAMETERS IWORK
    # to keep the same indices as in the Fortran code, the first component here is useless
    iwork = np.zeros((25,), dtype=np.int32)
    work  = np.zeros((25,), dtype=np.float64)

    if (bDenseJacobian) and (imas==0) and bTryToUseHessenberg:
      # the Jacobian can be transformed to Hessenberg, speeding up computations for large systems with dense Jacobians
      iwork[1] = 1
    else:
      iwork[1] = 0

    iwork[2] = nmax_step

    if max_ite_newton is not None:
      iwork[3] = max_ite_newton

    if bUseExtrapolatedGuess is not None:
      if not bUseExtrapolatedGuess:
        iwork[4] = 1

    # TODO: Radau5 requires that the variables be sorted by differentiation index for DAEs
    if imas==1:
      if var_index is not None: # DAE system of index>1 (index 1 does not need extra treatments)
       # n_index0 = np.count_nonzero(var_index == 0)
       # n_index1 = np.count_nonzero(var_index == 1)
       # n_index2 = np.count_nonzero(var_index == 2)
       # n_index3 = np.count_nonzero(var_index == 3)
       # assert n_index0 + n_index1 + n_index2 + n_index3 == neq, "Are some variables of index higher than 3 ?"
       # assert np.all(np.diff(var_index)>=0), 'components mus be sorted by index'
       # iwork[5] = n_index0 + n_index1 # number of index-0 and index-1 variables
       # iwork[6] = n_index2
       # iwork[7] = n_index3
        pass
        assert  var_index.size == y0.size
        if var_index.dtype != np.int32:
            var_index = var_index.astype(np.int32)#, casting='safe')
      else:
         var_index = np.zeros((y0.size,), dtype=np.int32)

    if bUsePredictiveController is not None:
      if bUsePredictiveController:
        iwork[8] = 1 # advanced time step controller of Gustafson
      else:
        iwork[8] = 2 # classical time step controller

    # for second-order systems #TODO: enable that !!!
    iwork[9]  = 0
    iwork[10] = 0

    # work is set to 0 for default behaviour
    #work[1] = 1e-16 # rounding unit
    if safetyFactor is not None:
      work[2] = safetyFactor
    if jacobianRecomputeFactor is not None:
      work[3] = jacobianRecomputeFactor
    if newton_tol is not None:
      work[3] = newton_tol
    if deadzone is not None: # deadzone for time step
      work[5] = deadzone[0]
      work[6] = deadzone[1]

    if max_step is not None:
      work[7] = max_step

    if step_evo_factor_bounds is not None: # min and max relative time step variation
      work[8] = step_evo_factor_bounds[0]
      work[9] = step_evo_factor_bounds[1]

    if bDebug:
      print('iwork = ', iwork)
      print('work = ', work)

    if bAlwaysApply2ndEstimate:
        nAlwaysUse2ndErrorEstimate = 1
    else:
        nAlwaysUse2ndErrorEstimate = 0

    if t_eval is None:
      iout = 1 # all time steps will be exported
    else:
      if len(t_eval)==1 and t_eval[-1]==tend:
        iout = 0
      else:
        raise NotImplemented('Non trivial values of `t_eval` are not yet supported')

    #################################
    ##  Call C-interface to Radau5 ##
    #################################
    if bDebug: print('Calling C interface')
    import sys; sys.stdout.flush()
    iwork = iwork[1:]
    work  = work[1:]
    c_radau5(tini, tend, first_step,
             neq, y0, yn,
             callable_fcn, callable_mass_fcn, callable_solout, callable_reportfun, nReport,
             rtol, atol,
             mljac, mujac,
             imas, mlmas, mumas, var_index,
             iwork, work,
             iout, info,
             bPrint, nMaxBadIte, nAlwaysUse2ndErrorEstimate)

    ################
    ##  Finalise  ##
    ################
    if iout == 1: # whole solution is exported
        tsol = np.array(tsol)
        ysol = np.array(np.array(ysol).T, order='F')
    else: # only final point
        tsol = tend
        ysol = yn

    out = radau_result(ysol, tsol)
    out.nfev   = info[0]  # number of function evaluations
    out.njev   = info[1]  # number of jacobian evaluations
    out.nstep  = info[2]  # number of computed steps
    out.naccpt = info[3]  # number of accepted steps
    out.nrejct = info[4]  # number of rejected steps
    out.ndec   = info[5]  # number of lu-decompositions
    out.nsol   = info[6]  # number of forward-backward substitutions
    out.nfail  = info[7]  # number of failed steps (Newton's fault)
    out.nlinsolves_err  = info[8]  # number of linear solves for error estimation
    IDID       = info[9] # exit code

    if bDebug:
        print('info=',info)

    out.success = (IDID > 0)
    if IDID== 1:  out.msg='COMPUTATION SUCCESSFUL'
    if IDID== 2:  out.msg='COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)'
    if IDID==-1:  out.msg='INPUT IS NOT CONSISTENT'
    if IDID==-2:  out.msg='LARGER NMAX IS NEEDED'
    if IDID==-3:  out.msg='STEP SIZE BECOMES TOO SMALL'
    if IDID==-4:  out.msg='MATRIX IS REPEATEDLY SINGULAR'

    if bReport:
      for key in reports.keys():
        reports[key] = np.array(reports[key])
        out.reports = reports
    if bPrint:
      print(f'IDID={IDID}, msg={out.msg}')
    return out
