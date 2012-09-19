/******************************************************************************
  Copyright (c) 2006,2007 by Turku PET Centre

  lss_lib.h

  Header file for some algorithms needed by nnls, ldp, lsi, lse, nlls etc

  Version:
  2006-07: Pauli Sundberg
  2007-05-17 Vesa Oikonen
    external DEBUG changed to local LSS_TEST.
    

******************************************************************************/
#ifndef _LSS_H12_H
#define _LSS_H12_H
/*****************************************************************************/

#include "memory_handler.h"

extern int _lss_h12(int mode, int lpivot, int l1, int m, double *u, int iue,
              double *up, double *cm, int ice, int icv, int ncv);


extern void _lss_g1(double a, double b, double *cterm, double *sterm, double *sig);

extern int _lss_hfti(double* A, int m_a, int m, int n_a, int n, double* B, int m_b, int n_b, 
               double tau, int* krank, double* rnorm, double* w1, 
               double* w2, int* ip);

extern void _lss_print_matrix(const double* M, int m, int n);
extern void _lss_print_matrix2(const double* M, int m, int n, int icv, int ice);
extern void _lss_matrix_times_vector( const double* M, int m, int n, int m_t, int n_t, const double* v,
                               int ie, double* x);
                               
extern double _lss_compare_matrix(const double* A, const double* B, int m, int n);


// global char to define debug level
int LSS_TEST;

/**
 Simple define to allocate workspace with memory handler
 (which must be initialized!).
 
 calls 'return 2;' if out of memory.
*/
#define GET_WORKSPACE(x,y,z) x = (y*)allocate_memory( (z) * sizeof(y) ); if (x == NULL) return 2;
#define FREE_WORKSPACE(x) free_memory(x);
#define SMALLEST_NON_ZERO 0.000001

#define ABS(a)      (((a) < 0) ? -(a) : (a))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
         
/*****************************************************************************/
#endif
