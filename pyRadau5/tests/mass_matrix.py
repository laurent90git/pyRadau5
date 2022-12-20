from pyRadau5 import integration
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.integrate

n = 100
x = np.linspace(0,1,n)
y0 = (np.exp((x-0.5)**2) - 1)/(np.exp(0.5**2) - 1)
dx = x[1]-x[0]
tf = 0.02
Tdirich = 1.

def testBandedMassMatrix(bPlot=False):
  
  # test slightly perturbed mass matrix
  mass_matrix = scipy.sparse.diags(diagonals=(-4e-1,1,-3e-1), offsets=[-1,0,1], shape=(n,n)).toarray(); mumas=1;mlmas=1;
  #mass_matrix = np.eye(n); mumas=mlmas=0
  minv = np.linalg.inv(mass_matrix)
  
  # mass_matrix = np.eye(n)
  # mumas=mlmas=n-4
  # mumas=mlmas=0

  def heat_modelfun(t,y,mode=None):
      """ Simple finite-difference discretisation of the heat equation,
          with left Neumann BC and right Dirichlet BC """
      dxdt = np.empty_like(y)
      dxdt[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dx**2
      dxdt[0]  =  (y[1] - y[0])   / dx**2
      dxdt[-1] =  (y[-2] - 2*y[-1]+ Tdirich) / dx**2
      if mode==1: # My' = f(y)
        return dxdt
      elif mode==2: # y' = (M^-1) f(y)
        return minv @ dxdt
      else:
        raise Exception('bad mode')

  # Compute solution with Radau5 for My' = f(y)
  outMass = integration.radau5(tini=0., tend=tf, y0=y0,
                              fun=lambda t,x: heat_modelfun(t,x,mode=1),
                              mljac=n, mujac=n,
                              mlmas=mlmas, mumas=mumas,
                              rtol=1e-8, atol=1e-8,
                              t_eval=None,
                              nmax_step = 100,
                              max_step = tf,
                              max_ite_newton=None, bUseExtrapolatedGuess=None,
                              bUsePredictiveController=None, safetyFactor=None,
                              deadzone=None, step_evo_factor_bounds=None,
                              jacobianRecomputeFactor=None, newton_tol=None,
                              mass_matrix=mass_matrix, var_index=None,
                              bPrint=False)
  assert outMass.success
  
  # Compute reference solution for y' = (M^-1)*F(y)
  outRef = scipy.integrate.solve_ivp(fun=lambda t,x: heat_modelfun(t,x,mode=2),
                                     y0=y0, t_span=(0,tf), rtol=1e-8, atol=1e-8,
                                     method='LSODA', uband=5, lband=5)
  assert outRef.success
   
  if bPlot:
      plt.figure()
      for it in range(len(outMass.t)):
          plt.plot(x, outMass.y[:,it])
      plt.grid()
      plt.xlabel('x')
      plt.ylabel('T')
      plt.title('Radau5 solution')
      
      plt.figure()
      for it in range(len(outRef.t)):
          plt.plot(x, outRef.y[:,it])
      plt.grid()
      plt.xlabel('x')
      plt.ylabel('T')
      plt.title('Reference solution')

  assert np.allclose(outRef.y[:,-1], outMass.y[:,-1], rtol=1e-5, atol=1e-5), 'Solution with banded mass matrix seems wrong'

def testDenseMassMatrix(bPlot=False):
  laplacien = scipy.sparse.diags(diagonals=(1,-2,1), offsets=[-1,0,1], shape=(n,n)).toarray()
  laplacien[0,0]=-1
  b = np.zeros(n,)
  b[-1] = Tdirich
  mass_matrix = np.linalg.inv(laplacien)
  heat_modelfun = lambda t,y: (1/dx**2) * (  y + mass_matrix @ b )
  heat_modelfun_no_mass = lambda t,y: (1/dx**2) * laplacien @ (  y + mass_matrix @ b )
  mlmas=mumas=None
  
  # Compute solution with Radau5 for My' = f(y)
  outMass = integration.radau5(tini=0., tend=tf, y0=y0,
                              fun=heat_modelfun,
                              mljac=n, mujac=n,
                              mlmas=mlmas, mumas=mumas,
                              rtol=1e-8, atol=1e-8,
                              t_eval=None,
                              nmax_step = 100,
                              max_step = tf,
                              max_ite_newton=None, bUseExtrapolatedGuess=None,
                              bUsePredictiveController=None, safetyFactor=None,
                              deadzone=None, step_evo_factor_bounds=None,
                              jacobianRecomputeFactor=None, newton_tol=None,
                              mass_matrix=mass_matrix, var_index=None,
                              bPrint=False)
  assert outMass.success
  
  # Compute reference solution for y' = (M^-1)*F(y)
  outRef = scipy.integrate.solve_ivp(fun=heat_modelfun_no_mass,
                                     y0=y0, t_span=(0,tf), rtol=1e-8, atol=1e-8,
                                     method='LSODA', uband=5, lband=5)
  assert outRef.success
   
  if bPlot:
      plt.figure()
      for it in range(len(outMass.t)):
          plt.plot(x, outMass.y[:,it])
      plt.grid()
      plt.xlabel('x')
      plt.ylabel('T')
      plt.title('Radau5 solution')
      
      plt.figure()
      for it in range(len(outRef.t)):
          plt.plot(x, outRef.y[:,it])
      plt.grid()
      plt.xlabel('x')
      plt.ylabel('T')
      plt.title('Reference solution')

  assert np.allclose(outRef.y[:,-1], outMass.y[:,-1], rtol=1e-5, atol=1e-5), 'Solution with banded mass matrix seems wrong'

  

if __name__=='__main__':
  testBandedMassMatrix(bPlot=True)
  testDenseMassMatrix(bPlot=True)