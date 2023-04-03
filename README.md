# pyRadau5

This package provides a Python interface to the well known stiff ODE/DAE Fortran integrator Radau5 developped by Hairer & Wanner [1,2,3].
The original Fortran code has been modified to improve its robustness on DAE problems (TODO: list of change somewhere). It is also possible to have a precise log output from the solver to investigate its behaviour.

This package has been developed by Laurent François, starting from an early interface developed by Laurent Series.

## Requirements
* GFortran compiler. I currently have not taken time to use other compilers.

## TODO
* Compatibility with Windows
* Event handling
* Secondary variables
* Performance comparison
* Make the interface as close to that of `scipy.integrate.solve_ivp` as possible

## Installation

Download the repository. Inside its folder, run the following command to install the package:

```bash
python setup.py install
```

Alternatively, you can install it in development mode:

```bash
python setup.py develop
```
which will enable you to use modify the code without having to reinstall the package after every modification.

## Usage

```python
import pyRadau5
import numpy as np
import matplotlib.pyplot

# Define the ODE system
# Curtiss-Hirschfelder problem
def odefun(t,y):
  return k*(y-np.cos(t))

# call the Python interface to Radau5
sol = pyRadau5.integrate(fun=odefun, t_span=(0.,5.), y0=[2.], ...)

# Plot the solution
plt.figure()
plt.plot(sol.t, sol.y[0,:], label='sol')
plt.plot(sol.t, np.cos(t), linestyle=':', label=r'$cos(t)$')
plt.grid()
plt.legend()
plt.xlabel('t')
plt.ylabel('y')
```
TODO: corresponding picture

## Examples

TODO

## Performance

TODO: comparison with Scipy and others

## Contributing

Pull requests are welcome. Please open an issue first to discuss what you would like to change.

## Bibliography
[1] Hairer, E., Lubich, C., & Roche, M. (2006). *The numerical solution of differential-algebraic systems by Runge-Kutta methods* (Vol. 1409), Springer

[2] Wanner, G., & Hairer, E. (1996). *Solving ordinary differential equations II* (Vol. 375), Springer

[3] Personal software webpage of Ernst Hairer: [http://www.unige.ch/~hairer/software.html](http://www.unige.ch/~hairer/software.html)

## Licence
The original Fortran code is under the [following license](http://www.unige.ch/~hairer/prog/licence.txt).
This package
