from integration.rock_radau import integration
import numpy as np
import matplotlib.pyplot as plt
n = 100
x = np.linspace(0,1,n)
y0 = (np.exp((x-0.5)**2) - 1)/(np.exp(0.5**2) - 1)
dx = x[1]-x[0]
tf = 0.02
Tdirich = 1.

if 0:
    if 1:
#        def heat_modelfun(t,y):
#            """ Simple finite-difference discretisation of the heat equation,
#                with Neumann BCs """
#            dxdt = np.empty_like(y)
#            dxdt[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
#            dxdt[0]  =  (y[1] - y[0])   / dx**2
#            dxdt[-1] =  (y[-2] - y[-1]) / dx**2
#            return dxdt
        def heat_modelfun(t,y):
            """ Simple finite-difference discretisation of the heat equation,
                with left Neumann BC and right Dirichlet BC """
            dxdt = np.empty_like(y)
            dxdt[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
            dxdt[0]  =  (y[1] - y[0])   / dx**2
            dxdt[-1] =  (y[-2] - 2*y[-1]+ Tdirich) / dx**2
            return dxdt
        
    else:
        #laplacien = np.diags((1,-2,1), shape=(n,n))
        import scipy.sparse
        laplacien = scipy.sparse.diags(diagonals=(1,-2,1), offsets=[-1,0,1], shape=(n,n)).toarray()
        laplacien[0,0]=-1
        def heat_modelfun(t,y):
            dxdt = laplacien @ y
            dxdt[-1] +=Tdirich
            return (1/dx**2) * dxdt

    mass_matrix = np.eye(n)
    #mumas=mlmas=n-4
    mumas=mlmas=0
    #mass_matrix = np.eye(n) + np.random.random((n,n))*1e-2
    #mass_matrix[n-1,0]=0
    #mass_matrix[0,n-1]=0
    #mass_matrix = np.ones((n,n))
    #mass_matrix = None
    
else: # test with dense mass matrix    
    import scipy.sparse
    laplacien = scipy.sparse.diags(diagonals=(1,-2,1), offsets=[-1,0,1], shape=(n,n)).toarray()
    laplacien[0,0]=-1
    b = np.zeros(n,)
    b[-1] = Tdirich
    mass_matrix = np.linalg.inv(laplacien)
    heat_modelfun = lambda t,y: (1/dx**2) * (  y + mass_matrix @ b )
    mlmas=mumas=None

#def fcn_rock(n, t, y, ydot, *args):
    #y_np = np.ctypeslib.as_array(y, shape=(n[0],))
    #ydot_np = modeleB_odefun(t[0], y_np, options)
    #for i in range(n[0]):
#      ydot[i] = ydot_np[i]

if 1:
    out = integration.radau5(tini=0., tend=tf, yini=y0,
                            fun=heat_modelfun,
                            mljac=n, mujac=n,
                            mlmas=mlmas, mumas=mlmas,
                            rtol=1e-4, atol=1e-4,
                            t_eval=None,
                            nmax_step = 100000,
                            max_step = tf,
                            max_ite_newton=None, bUseExtrapolatedGuess=None,
                            bUsePredictiveController=None, safetyFactor=None,
                            deadzone=None, step_evo_factor_bounds=None,
                            jacobianRecomputeFactor=None, newton_tol=None,
                            mass_matrix=mass_matrix, var_index=None)
            
    print('nfev={}, njev={}, nlu={}, linsolve={}'.format(out.nfev, out.njev, out.ndec, out.nsol))
else:
    import scipy.integrate
    out = scipy.integrate.solve_ivp(fun=heat_modelfun, y0=y0, t_span=(0,tf), rtol=1e-6, atol=1e-6,
                               method='LSODA', uband=5, lband=5)
#ROCK4, solveur explicite (non fonctionnel)
#out = integration.rock4(t_vec=np.linspace(0,tf,2),
#                        yini=y0,
#                        fcn=lambda n, t, y, ydot: fcn_rock(n, t, y, ydot, options=options),
#                        tol=tol)
#


plt.figure()
for it in range(len(out.t)):
    plt.plot(x, out.y[:,it])
plt.grid()
plt.xlabel('x')
plt.ylabel('T')
plt.show()
