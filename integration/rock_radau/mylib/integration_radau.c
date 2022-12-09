#include<stdio.h>

#include "integration_radau.h"

void radau5_integration(double tini, double tend, int n, double *yini, double *y, func_radau fcn,
		        func_solout_radau solout, double rtol, double atol, int njac, int iout, int *info)
{
  // both rtol and atol are scalars
  int itol=0;

  // jacobian is computed internally by finite differences
  int ijac=0;
  // lower bandwidth of jacobian 
  int mljac = njac;
  // upper bandwidth of jacobian 
  int mujac = mljac; 

  // mass matrix (assumed to be the identity matrix)
  int imas=0;
  int mlmas;
  int mumas;

  // output routine is used during integration
  ////int iout=1;

  // size of array work 
  int ljac=mljac+mujac+1;
  int le=2*mljac+mujac+1;
  int lmas=0;
  int lwork = n*(ljac+lmas+3*le+12)+20;
  double work[lwork];

  // size of array lwork
  int liwork = 3*n+20;
  int iwork[liwork];

  // real and integer parameters
  double rpar;
  int ipar;

  // integer returning the success of the integration
  int idid;

  // initial time t and initial time step  dt 
  double t=tini;
  double dt=0.;

  int i;

  // initial solution
  for (i=0; i<n; ++i) y[i] = yini[i];

  for(i=0; i<20; i++)
  {
    iwork[i]=0;
    work[i]=0.0;
  }

  // directly calling fortran
  radau5(&n, fcn, &t, y, &tend, &dt,
         &rtol, &atol, &itol,
         jac_radau, &ijac, &mljac, &mujac,
         mas_radau, &imas, &mlmas, &mumas,
         solout, &iout,
         work, &lwork, iwork, &liwork,
         &rpar, &ipar, &idid);

  // save & print statistics
  info[0] = iwork[13];  
  info[1] = iwork[14];  
  info[2] = iwork[15]; 
  info[3] = iwork[16];  
  info[4] = iwork[17];  
  info[5] = iwork[18];  
  info[6] = iwork[19];  
}

void jac_radau(int *n, double *x, double *y, double *dfy, int *ldfy, double *rpar, double *ipar)
{
}


void mas_radau(int *n,double *am, int *lmas,int *rpar, int *ipar)
{
}
