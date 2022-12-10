from integration.rock_radau import integration
import numpy as np
import matplotlib.pyplot as plt
n = 100
x = np.linspace(0,1,n)
y0 = (np.exp((x-0.5)**2) - 1)/(np.exp(0.5**2) - 1)
dx = x[1]-x[0]
tf = 1.2

def heat_modelfun(t,y):
    """ Simple finite-difference discretisation of the heat equation,
        with Neumann BCs """
    dxdt = np.empty_like(y)
    dxdt[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
    dxdt[0]  =  (y[1] - y[0])   / dx**2
    dxdt[-1] =  (y[-2] - y[-1]) / dx**2
    return dxdt
    
#mass_matrix = np.eye(n)
mass_matrix = np.ones((n,n))

def fcn_rock(n, t, y, ydot, *args):
    y_np = np.ctypeslib.as_array(y, shape=(n[0],))
    ydot_np = modeleB_odefun(t[0], y_np, options)
    for i in range(n[0]):
      ydot[i] = ydot_np[i]


out = integration.radau5(tini=0., tend=tf, yini=y0,
                        fun=heat_modelfun,
                        mujac=n, mljac=n,
                        rtol=1e-4, atol=1e-4,
                        t_eval=None,
                        nmax_step = 100000000000,
                        max_step = tf,
                        max_ite_newton=None, bUseExtrapolatedGuess=None,
                        bUsePredictiveController=None, safetyFactor=None,
                        deadzone=None, step_evo_factor_bounds=None,
                        jacobianRecomputeFactor=None, newton_tol=None,
                        mass_matrix=mass_matrix, var_index=None)
        
print('nfev={}, njev={}, nlu={}, linsolve={}'.format(out.nfev, out.njev, out.ndec, out.nsol))

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
