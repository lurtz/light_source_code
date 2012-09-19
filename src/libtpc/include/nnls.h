/******************************************************************************
  Copyright (c) 2002,2003,2007,2009 by Turku PET Centre

  nnls.h
  
  Version:
  2002-08-19 Vesa Oikonen
  2003-05-08 Kaisa Sederholm & VO
  2003-05-12 KS
  2007-05-17 VO
  2009-04-27 VO

******************************************************************************/
#ifndef _NNLS_H
#define _NNLS_H
/*****************************************************************************/
int NNLS_TEST;
/*****************************************************************************/
extern int nnls(double **a, int m, int n, double *b, double *x,
          double *rnorm, double *w, double *zz, int *index);
extern int nnlsWght(int N, int M, double **A, double *b, double *weight);
extern int nnlsWghtSquared(int N, int M, double **A, double *b, double *sweight);
/*****************************************************************************/
#endif
