#include<stdio.h>

#include "integration_radau.h"
//void radau5_integration(double tini, double tend, int n, double *yini, double *y, func_radau fcn, func_mas_radau mas_fcn, func_solout_radau solout, double rtol, double atol, int mljac, int mujac, int imas, int mlmas, int mumas, int *iwork_in, double *work_in, int iout, int *info)
void radau5_integration(double tini, double tend, double first_step,
                        int n, // size of the system
                        double *yini, // pointer to the initial solution vector
                        double *y, //
                        func_radau fcn, // interface to the Python time derivative function
                        func_mas_radau mas_fcn, // mass matrix evaluation function
                        func_solout_radau solout, // solution export function
                        double rtol, double atol, // error tolerances (scalar only)
                        int mljac, int mujac, // Jacobian lower and upper bandwiths
                        int imas, int mlmas, int mumas, // Mass matrix lower and upper bandwiths
                        int *iwork_in, // integer parameters
                        double *work_in, // decimal parameters
                        int iout,  // solution export mode
                        int *info) // statistics
{
  int bPrint=0;
  if (bPrint) {  
    printf("n=%i, rtol=%f, atol=%f\n", n, rtol, atol);
    printf("mljac=%i, mujac=%i\n", mljac, mujac);
    printf("imas=%i, mlmas=%i, mumas=%i\n", imas, mlmas, mumas);
    printf("iout=%i\n", iout);
  }
  
  // both rtol and atol are scalars
  int itol=0; // tolerances are scalar
  int ijac=0; // jacobian is computed internally by finite differences
  //TODO: enable user-provided Jacobian function ?

  // size of array work 
  int ljac=mljac+mujac+1;
  int le=2*mljac+mujac+1;
  int lmas=mlmas+mumas+1;
  int lwork = n*(ljac+lmas+3*le+12)+20; // minimum size  
  int liwork = 3*n+20;
  // work arrays
  double work[lwork];
  int iwork[liwork];

  if (bPrint) { 
    printf("itol=%i, ijac=%i, ljac=%i, le=%i, lmas=%i\n", itol, ijac, ljac, le, lmas);
    printf("lwork=%i, liwork=%i\n", lwork, liwork);
  }
  
  // real and integer parameters
  double rpar;
  int ipar;

  // integer returning the success of the integration
  int idid;

  // initial time t and initial time step  dt 
  double t=tini;
  double dt=first_step;

  int i;
  // initial solution
  for (i=0; i<n; ++i) y[i] = yini[i];

  for(i=0; i<20; i++)
  {
    if (bPrint) { 
      printf("  iwork_in[%2i]  = %16i, \t work_in[%2i]  = %16f\n",i, iwork_in[i],i,work_in[i]);
    }
    iwork[i] = iwork_in[i];
    work[i]  =  work_in[i];
  }
  if (bPrint) printf("Calling radau from C interface\n");

  // directly calling fortran
  radau5(&n, fcn, &t, y, &tend, &dt,
         &rtol, &atol, &itol,
         jac_radau, &ijac, &mljac, &mujac,
         mas_fcn, &imas, &mlmas, &mumas,
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
  info[7] = idid;  
}

void jac_radau(int *n, double *x, double *y, double *dfy, int *ldfy, double *rpar, double *ipar)
{
}


/*void mas_radau(int *n,double *am, int *lmas,int *rpar, int *ipar)
{
}*/
