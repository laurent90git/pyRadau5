#ifndef INTEGRATION_RADAU_H
#define INTEGRATION_RADAU_H

typedef void(*func_radau)(int*, double*, double*, double*, double*, int*);

typedef void(*func_solout_radau)(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*);

void radau5_integration(double tini, double tend, int n, double *yini, double *y, func_radau fcn,
		        func_solout_radau solout, double rtol, double atol, int mljac, int iout, int *info);

void radau5(int *n, func_radau fcn, double *x, double *y, double *xend, double *h,
            double *rtol, double *atol, int *itol,
            void jac_radau(int*, double*, double*, double*, int*, double*, double*),
            int *ijac, int *mljac, int *mujac,
            void mas_radau(int *,double *, int *,int *, int *),
            int *imas, int *mlmas, int *mumas,
            func_solout_radau solout,
            int *iout,
            double *work, int *lwork,int *iwork, int *liwork,
            double *rpar, int *ipar, int *idid);

void jac_radau(int *n, double *x, double *y, double *dfy, int *ldfy, double *rpar, double *ipar);

void mas_radau(int *n,double *am, int *lmas,int *rpar, int *ipar);

#endif
