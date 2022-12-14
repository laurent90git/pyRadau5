#ifndef INTEGRATION_RADAU_H
#define INTEGRATION_RADAU_H

typedef void(*func_radau)(int*, double*, double*, double*, double*, int*);

typedef void(*func_mas_radau)(int*, double*, int*, int*, int*);

typedef void(*func_jac_radau)(int*, double*, double*, double*, int*, double*, double*);

typedef void(*func_solout_radau)(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*);
       

void radau5_integration(double tini, double tend, double first_step, int n, double *y0,
        double *y, func_radau fcn, func_mas_radau mas_fcn, func_solout_radau solout,
        double rtol, double atol, int mljac, int mujac, int imas, int mlmas, int mumas, int* var_index,
        int *iwork_in, double *work_in, int iout, int *info,
        int bPrint, int nMaxBadIte, int nAlwaysUse2ndErrorEstimate);

void radau5(int *n, func_radau fcn, double *x, double *y, double *xend, double *h,
            double *rtol, double *atol, int *itol,
            void jac_radau(int*, double*, double*, double*, int*, double*, double*),
            int *ijac, int *mljac, int *mujac,
            func_mas_radau mas_radau,
            int *imas, int *mlmas, int *mumas, int* var_index,
            func_solout_radau solout,
            int *iout,
            double *work, int *lwork,int *iwork, int *liwork,
            double *rpar, int *ipar, int *idid,
            int *bPrint, int* nMaxBadIte, int* nAlwaysUse2ndErrorEstimate);

void jac_radau(int *n, double *x, double *y, double *dfy, int *ldfy, double *rpar, double *ipar);

#endif
