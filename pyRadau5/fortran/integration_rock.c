#include <stdio.h>

#include "integration_rock.h"

void rock4_integration(double tini, double tend, int n, double *yini, double *y,
                       func_rock fcn, double tol, int *info)
{
  // time and time step
  double t, dt;

  //  required tolerance
  double rtol = tol;
  double atol = rtol;

  // iwork : integer array of length 12 that gives information
  //         on how the problem is to be solved
  int iwork[12];

  // workspace 
  double work[8*n];

  // report on successfulness upon return
  int idid;

  int i;

  // initialization of t, dt and y
  t = tini;
  dt = 1.e-6;
  for (i=0; i<n; ++i) y[i] = yini[i];

  // iwork[0]=0  rock4 attempts to compute the spectral radius internally
  // iwork[1]=1  the jacobian is constant 
  // iwork[2]=0  return solution at tend
  // iwork[3]=0  atol and rtol are scalars
  iwork[0] = 0;
  iwork[1] = 1;
  iwork[2] = 0;
  iwork[3] = 0;

  // directly calling fortran
  rock4(&n, &t, &tend, &dt , y, fcn, &atol, &rtol, work, iwork, &idid);

  // save statistics
  info[0] = iwork[4];    // number of function evaluations.
  info[1] = iwork[5];    // number of steps
  info[2] = iwork[6];    // number of accepted steps
  info[3] = iwork[7];    // number of rejectd steps
  info[4] = iwork[8];    // number of evaluations of f used to estimate the spectral radius
  info[5] = iwork[9];    // maximum number of stages used
  info[6] = iwork[10];   // maximum value of the estimated bound for the spectral radius
  info[7] = iwork[11];   // minimum value of the estimated bound for the spectral radius
}

void rock4_integration_dt(double tini, double tend, int n, double *yini, double *y,
                          func_rock fcn, double tol, double *tsol, int *info)
{
  // time and time step
  double t, dt;

  //  required tolerance
  double rtol = tol;
  double atol = rtol;

  // iwork : integer array of length 12 that gives information
  //         on how the problem is to be solved
  int iwork[12];

  // workspace 
  double work[8*n];

  // report on successfulness upon return
  int idid;
   
  int i, icmpt;

  // initialization of t, dt and y
  t = tini;
  dt = 1.e-6;
  for (i=0; i<n; ++i) y[i] = yini[i];

  // iwork[0]=0  rock4 attempts to compute the spectral radius internally
  // iwork[1]=1  the jacobian is constant 
  // iwork[2]=1  
  // iwork[3]=0  atol and rtol are scalars
  iwork[0] = 0;
  iwork[1] = 1;
  iwork[2] = 1;
  iwork[3] = 0;

  tsol[0] = tini;
  icmpt = 0;
  do
  {
    // directly calling fortran
    rock4(&n, &t, &tend, &dt , y, fcn, &atol, &rtol, work, iwork, &idid);
    //printf(" t = %lf\n", t);
    icmpt++;
    tsol[icmpt] = t;
  }
  while (idid==2);

  // save statistics
  info[0] = iwork[4];    // number of function evaluations.
  info[1] = iwork[5];    // number of steps
  info[2] = iwork[6];    // number of accepted steps
  info[3] = iwork[7];    // number of rejectd steps
  info[4] = iwork[8];    // number of evaluations of f used to estimate the spectral radius
  info[5] = iwork[9];    // maximum number of stages used
  info[6] = iwork[10];   // maximum value of the estimated bound for the spectral radius
  info[7] = iwork[11];   // minimum value of the estimated bound for the spectral radius
}

double rho(int *n, double *t, double y)
{
}
