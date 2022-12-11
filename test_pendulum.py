from integration.rock_radau import integration
import numpy as np
import matplotlib.pyplot as plt


#%% Setup the model based on the chosen formulation
def generateSytem(chosen_index, theta_0=np.pi/2, theta_dot0=0., r0=1., m=1, g=9.81):
    """ Generates the DAE function representing the pendulum with the desired
    index (0 to 3).
        Inputs:
          chosen_index: int
            index of the DAE formulation
          theta_0: float
            initial angle of the pendulum in radians
          theta_dot0: float
            initial angular velocity of the pendulum (rad/s)
          r0: float
            length of the rod
          m: float
            mass of the pendulum
          g: float
            gravitational acceleration (m/s^2)
        Outputs:
          dae_fun: callable
            function of (t,x) that represents the system. Will be given to the
            solver.
          jac_dae: callable
            jacobian of dae_fun with respect to x. May be given to the solver.
          mass: array_like
            mass matrix
          Xini: initial condition for the system
    """

    if chosen_index==3:
        def dae_fun(t,X):
          # X= (x,y,xdot=vx, ydot=vy, lbda)
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([vx,
                           vy,
                           -x*lbda/m,
                           -g-(y*lbda)/m,
                            x**2 + y**2 -r0**2])
        mass = np.eye(5) # mass matrix M
        mass[-1,-1]=0
        var_index = np.array([0,0,0,0,3])

        def jac_dae(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([[0.,0.,1.,0.,0.],
                           [0.,0.,0.,1.,0.],
                           [-lbda/m, 0., 0., 0., -x/m],
                           [0.,-lbda/m,  0., 0., -y/m],
                           [2*x, 2*y, 0., 0., 0.]])

    elif chosen_index==2:
        def dae_fun(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([vx,
                           vy,
                           -x*lbda/m,
                           -g-(y*lbda)/m,
                            x*vx+y*vy,])
        mass = np.eye(5)
        mass[-1,-1]=0
        var_index = np.array([0,0,0,0,2])

        def jac_dae(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([[0.,0.,1.,0.,0.],
                           [0.,0.,0.,1.,0.],
                           [-lbda/m, 0., 0., 0., -x/m],
                           [0.,-lbda/m,  0., 0., -y/m],
                           [vx, vy, x, y, 0.]])

    elif chosen_index==1:
        def dae_fun(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([vx,
                           vy,
                           -x*lbda/m,
                           -g-(y*lbda)/m,
                           lbda*(x**2+y**2)/m + g*y - (vx**2 + vy**2)])
        mass = np.eye(5)
        mass[-1,-1]=0
        var_index = np.array([0,0,0,0,1])
        def jac_dae(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          return np.array([[0.,0.,1.,0.,0.],
                           [0.,0.,0.,1.,0.],
                           [-lbda/m, 0., 0., 0., -x/m],
                           [0.,-lbda/m,  0., 0., -y/m],
                           [2*x*lbda/m, 2*y*lbda/m + g, -2*vx, -2*vy, (x**2+y**2)/m]])

    elif chosen_index==0:
        def dae_fun(t,X):
          x=X[0]; y=X[1]; vx=X[2]; vy=X[3]; lbda=X[4]
          dvx = -lbda*x/m
          dvy = -lbda*y/m - g
          rsq = x**2 + y**2 # = r(t)^2
          dt_lbda = (1/m)*(  -g*vy/rsq + 2*(vx*dvx+vy*dvy)/rsq  + (vx**2+vy**2-g*y)*(2*x*vx+2*y*vy)/(rsq**2))
          return np.array([vx,
                           vy,
                           dvx,
                           dvy,
                           dt_lbda])
        mass=None # or np.eye(5) the identity matrix
        var_index = np.array([0,0,0,0,0])
        jac_dae = None # use the finite-difference routine, this expression would
                       # otherwise be quite heavy :)
    else:
      raise Exception('index must be in [0,3]')

    # alternatively, define the Jacobian via finite-differences, via complex-step
    # to ensure machine accuracy
    if jac_dae is None:
        import scipy.optimize._numdiff
        jac_dae = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                                    fun=lambda x: dae_fun(t,x),
                                    x0=x, method='cs',
                                    rel_step=1e-50, f0=None,
                                    bounds=(-np.inf, np.inf), sparsity=None,
                                    as_linear_operator=False, args=(),
                                    kwargs={})
    ## Otherwise, the Radau solver uses its own routine to estimate the Jacobian, however the
    # original Scip routine is only adapted to ODEs and may fail at correctly
    # determining the Jacobian because of it would chosse finite-difference step
    # sizes too small with respect to the problem variables (<1e-16 in relative).
    # jac_dae = None

    ## Initial condition (pendulum at an angle)
    x0 =  r0*np.sin(theta_0)
    y0 = -r0*np.cos(theta_0)
    vx0 = r0*theta_dot0*np.cos(theta_0)
    vy0 = r0*theta_dot0*np.sin(theta_0)
    lbda_0 = (m*r0*theta_dot0**2 +  m*g*np.cos(theta_0))/r0 # equilibrium along the rod's axis

     # the initial condition should be consistent with the algebraic equations
    Xini = np.array([x0,y0,vx0,vy0,lbda_0])

    return dae_fun, jac_dae, mass, Xini, var_index


if __name__=='__main__':
    # Test the pendulum
    from scipy.integrate import solve_ivp
    from numpy.testing import (assert_, assert_allclose,
                           assert_equal, assert_no_warnings, suppress_warnings)
    import matplotlib.pyplot as plt

    ###### Parameters to play with
    chosen_index = 3 # The index of the DAE formulation
    tf = 10.0        # final time (one oscillation is ~2s long)
    rtol=1e-6; atol=rtol # relative and absolute tolerances for time adaptation
    # dt_max = np.inf
    # rtol=1e-3; atol=rtol # relative and absolute tolerances for time adaptation
    bPrint=False # if True, additional printouts from Radau during the computation
    bDebug=False # sutdy condition number of the iteration matrix


    ## Physical parameters for the pendulum
    theta_0=np.pi/6 # initial angle
    theta_dot0=0. # initial angular velocity
    r0=3.  # rod length
    m=1.   # mass
    g=9.81 # gravitational acceleration

    dae_fun, jac_dae, mass, Xini, var_index= generateSytem(chosen_index, theta_0, theta_dot0, r0, m, g)
    n = Xini.size
    # jac_dae = None
    #%% Solve the DAE
    print(f'Solving the index {chosen_index} formulation')
    sol = integration.radau5(tini=0., tend=tf, yini=Xini,
                        fun=dae_fun,
                        mljac=n, mujac=n,
                        mlmas=0, mumas=0,
                        rtol=1e-4, atol=1e-4,
                        t_eval=None,
                        nmax_step = 100000,
                        max_step = tf,
                        max_ite_newton=None, bUseExtrapolatedGuess=None,
                        bUsePredictiveController=None, safetyFactor=None,
                        deadzone=None, step_evo_factor_bounds=None,
                        jacobianRecomputeFactor=None, newton_tol=None,
                        mass_matrix=mass, var_index=var_index)

    print("DAE of index {} {} in {} time steps, {} fev, {} jev, {} LUdec".format(
          chosen_index, 'solved'*sol.success+(1-sol.success)*'failed',
          sol.t.size, sol.nfev, sol.njev, sol.ndec))

    # recover the time history of each variable
    x=sol.y[0,:]; y=sol.y[1,:]; vx=sol.y[2,:]; vy=sol.y[3,:]; lbda=sol.y[4,:]
    T = lbda * np.sqrt(x**2+y**2)
    theta= np.arctan(x/y)

    #%% Compute true solution (ODE on the angle in polar coordinates)
    def fun_ode(t,X):
      theta=X[0]; theta_dot = X[1]
      return np.array([theta_dot,
                       -g/r0*np.sin(theta)])

    Xini_ode= np.array([theta_0,theta_dot0])
    sol_ode = solve_ivp(fun=fun_ode, t_span=(0., tf), y0=Xini_ode,
                    rtol=rtol, atol=atol, max_step=tf/10, method='DOP853',
                    dense_output=True)
    theta_ode = sol_ode.y[0,:]
    theta_dot = sol_ode.y[1,:]
    x_ode =  r0*np.sin(theta_ode)
    y_ode = -r0*np.cos(theta_ode)
    vx_ode =  r0*theta_dot*np.cos(theta_ode)
    vy_ode =  r0*theta_dot*np.sin(theta_ode)
    T_ode = m*r0*theta_dot**2 + m*g*np.cos(theta_ode)

    #%% Compare the DAE solution and the true solution
    plt.figure()
    plt.plot(sol.t,x, color='tab:orange', linestyle='-', label=r'$x_{DAE}$')
    plt.plot(sol_ode.t,x_ode, color='tab:orange', linestyle='--', label=r'$x_{ODE}$')
    plt.plot(sol.t,y, color='tab:blue', linestyle='-', label=r'$y_{DAE}$')
    plt.plot(sol_ode.t,y_ode, color='tab:blue', linestyle='--', label=r'$y_{ODE}$')
    plt.plot(sol.t,y**2+x**2, color='tab:green', label=r'$x_{DAE}^2 + y_{DAE}^2$')
    plt.grid()
    plt.legend()
    plt.title('Comparison with the true solution')

    #%% Analyze constraint violations
    # we check how well equations (5), (6) and (7) are respected
    fig,ax = plt.subplots(3,1,sharex=True)
    constraint = [None for i in range(3)]
    constraint[2] = x**2 + y**2 - r0**2 #index 3
    constraint[1] = x*vx+y*vy #index 12
    constraint[0] = lbda*(x**2+y**2)/m + g*y - (vx**2 + vy**2) #index 1
    for i in range(len(constraint)):
      ax[i].plot(sol.t, constraint[i])
      # ax[i].semilogx(sol.t, np.abs(constraint[i]))
      ax[i].grid()
      ax[i].set_ylabel('index {}'.format(i+1))
    ax[-1].set_xlabel('t (s)')
    fig.suptitle('Constraints violation')


    #%% Plot the solution and some useful statistics
    fig, ax = plt.subplots(5,1,sharex=True, figsize=np.array([1.5,3])*5)
    i=0
    ax[i].plot(sol.t, x,     color='tab:orange', linestyle='-', linewidth=2, marker='.', label='x')
    ax[i].plot(sol.t, y,     color='tab:blue', linestyle='-', linewidth=2, marker='.', label='y')
    ax[i].plot(sol_ode.t, x_ode,     color='tab:orange', linestyle='--', linewidth=2, marker=None, label='x ODE')
    ax[i].plot(sol_ode.t, y_ode,     color='tab:blue', linestyle='--', linewidth=2, marker=None, label='y ODE')
    ax[i].set_ylim(-1.2*r0, 1.2*r0)
    ax[i].legend(frameon=False)
    ax[i].grid()
    ax[i].set_ylabel('positions')
    
    plt.show()
