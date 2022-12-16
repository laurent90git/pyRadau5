from integration.rock_radau import integration
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

n = 100
x = np.linspace(0,1,n)
y0 = (np.exp((x-0.5)**2) - 1)/(np.exp(0.5**2) - 1)
dx = x[1]-x[0]
tf = 0.02
Tdirich = 1.

if 1: # test with banded mass matrix
    # test slightly perturbed mass matrix
    mass_matrix = scipy.sparse.diags(diagonals=(-4e-1,1,-3e-1), offsets=[-1,0,1], shape=(n,n)).toarray(); mumas=1;mlmas=1;
    #mass_matrix = np.eye(n); mumas=mlmas=0
    minv = np.linalg.inv(mass_matrix)
    
    # mass_matrix = np.eye(n)
    # # mumas=mlmas=n-4
    # mumas=mlmas=0

    def heat_modelfun(t,y):
        """ Simple finite-difference discretisation of the heat equation,
            with left Neumann BC and right Dirichlet BC """
        dxdt = np.empty_like(y)
        dxdt[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
        dxdt[0]  =  (y[1] - y[0])   / dx**2
        dxdt[-1] =  (y[-2] - 2*y[-1]+ Tdirich) / dx**2
        # return minv @ dxdt
        return dxdt


else: # test with dense mass matrix
    import scipy.sparse
    laplacien = scipy.sparse.diags(diagonals=(1,-2,1), offsets=[-1,0,1], shape=(n,n)).toarray()
    laplacien[0,0]=-1
    b = np.zeros(n,)
    b[-1] = Tdirich
    mass_matrix = np.linalg.inv(laplacien)
    heat_modelfun = lambda t,y: (1/dx**2) * (  y + mass_matrix @ b )
    mlmas=mumas=None

#print('mass_matrix=\n', mass_matrix)

if 1:
    out = integration.radau5(tini=0., tend=tf, y0=y0,
                            fun=heat_modelfun,
                            mljac=n, mujac=n,
                            mlmas=mlmas, mumas=mumas,
                            rtol=1e-4, atol=1e-4,
                            t_eval=None,
                            nmax_step = 100,
                            max_step = tf,
                            max_ite_newton=None, bUseExtrapolatedGuess=None,
                            bUsePredictiveController=None, safetyFactor=None,
                            deadzone=None, step_evo_factor_bounds=None,
                            jacobianRecomputeFactor=None, newton_tol=None,
                            mass_matrix=mass_matrix, var_index=None,
                            bPrint=False)

    print('nfev={}, njev={}, nlu={}, linsolve={}'.format(out.nfev, out.njev, out.ndec, out.nsol))
else:
    # assert mass_matrix is None, 'LSODA is not comaptible with a mass matrix'
    import scipy.integrate
    out = scipy.integrate.solve_ivp(fun=heat_modelfun, y0=y0, t_span=(0,tf), rtol=1e-4, atol=1e-4,
                               method='LSODA', uband=5, lband=5)

plt.figure()
for it in range(len(out.t)):
    plt.plot(x, out.y[:,it])
plt.grid()
plt.xlabel('x')
plt.ylabel('T')
plt.show()
