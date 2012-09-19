/*****************************************************************************

  nnls.c  (c) 2002-2009 Turku PET Centre

  This file contains the routine NNLS (nonnegative least squares)
  and the subroutines required by it, except _lss_h12, which is in
  file 'lss_h12.c'.
  
  This routine is based on the text and fortran code in
  C.L. Lawson and R.J. Hanson, Solving Least Squares Problems,
  Prentice-Hall, Englewood Cliffs, New Jersey, 1974.

  Version:
  2002-08-19 Vesa Oikonen
  2003-05-08 Kaisa Sederholm & VO
    Included function nnlsWght().
  2003-05-12 KS
    Variable a_dim1 excluded
    Usage of the coefficient matrix altered so that it is
    given in a[][] instead of a[].
  2003-11-06 VO
    If n<2, then itmax is set to n*n instead of previous n*3.
  2004-09-17 VO
    Doxygen style comments.
  2006-24-04 Pauli Sundberg
    Renamed function _nnls_h12(..) to _lss_h12(..) and moved it to separeted
    file (lss_lib.c) since it's used by lse.c too. Same for g1 and g2.
  2006-24-06 Pauli Sundberg
    Added some debuging output, and made some comments more precise.
  2007-05-17 VO
    DEBUG changed to NNLS_TEST.
  2009-04-16 VO
    Corrected a bug in nnls() which may have caused an infinite loop.
  2009-04-27 VO
    Added function nnlsWghtSquared() for faster pixel-by-pixel calculations.
    Checking for exceeding iteration count is corrected in nnls().

    
*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/****************************************************************************/
#include "include/nnls.h"
#include "include/lss_lib.h"
/****************************************************************************/
/* Local function definitions */
/****************************************************************************/

/****************************************************************************/
/** Algorithm NNLS (Non-negative least-squares)
 *
 *  Given an m by n matrix A, and an m-vector B, computes an n-vector X,
 *  that solves the least squares problem
 *      A * X = B   , subject to X>=0
 *
 *  Instead of pointers for working space, NULL can be given to let this
 *  function to allocate and free the required memory.
 *
\return Function returns 0 if succesful, 1, if iteration count exceeded 3*N,
 *  or 2 in case of invalid problem dimensions or memory allocation error.
 */
int nnls(
  /** On entry, a[ 0... N ][ 0 ... M ] contains the M by N matrix A.
   *  On exit, a[][] contains the product matrix Q*A, where Q is an m by n
   *  orthogonal matrix generated implicitly by this function.*/
  double **a,
  /** Matrix dimension m */
  int m,
  /** Matrix dimension n */
  int n,
  /** On entry, b[] must contain the m-vector B.
      On exit, b[] contains Q*B */
  double *b,
  /** On exit, x[] will contain the solution vector */
  double *x,
  /** On exit, rnorm contains the Euclidean norm of the residual vector.
   *  If NULL is given, no rnorm is calculated */
  double *rnorm,
  /** An n-array of working space, wp[].
      On exit, wp[] will contain the dual solution vector.
      wp[i]=0.0 for all i in set p and wp[i]<=0.0 for all i in set z. 
      Can be NULL, which causes this algorithm to allocate memory for it. */
  double *wp,
  /** An m-array of working space, zz[]. 
      Can be NULL, which causes this algorithm to allocate memory for it. */
  double *zzp,
  /** An n-array of working space, index[]. 
      Can be NULL, which causes this algorithm to allocate memory for it. */
  int *indexp
) {
  int pfeas, ret=0, iz, jz, iz1, iz2, npp1, *index;
  double d1, d2, sm, up, ss, *w, *zz;
  int iter, k, j=0, l, itmax, izmax=0, nsetp, ii, jj=0, ip;
  double temp, wmax, t, alpha, asave, dummy, unorm, ztest, cc;


  /* Check the parameters and data */
  if(m<=0 || n<=0 || a==NULL || b==NULL || x==NULL) return(2);
  /* Allocate memory for working space, if required */
  if(wp!=NULL) w=wp; else w=(double*)calloc(n, sizeof(double));
  if(zzp!=NULL) zz=zzp; else zz=(double*)calloc(m, sizeof(double));
  if(indexp!=NULL) index=indexp; else index=(int*)calloc(n, sizeof(int));
  if(w==NULL || zz==NULL || index==NULL) return(2);

  /* Initialize the arrays INDEX[] and X[] */
  for(k=0; k<n; k++) {x[k]=0.; index[k]=k;}
  iz2=n-1; iz1=0; nsetp=0; npp1=0;

  /* Main loop; quit if all coeffs are already in the solution or */
  /* if M cols of A have been triangularized */
  iter=0; if(n<3) itmax=n*3; else itmax=n*n;
  while(iz1<=iz2 && nsetp<m) {
    /* Compute components of the dual (negative gradient) vector W[] */
    for(iz=iz1; iz<=iz2; iz++) {
      j=index[iz]; sm=0.; for(l=npp1; l<m; l++) sm+=a[j][l]*b[l];
      w[j]=sm;
    }

    if (NNLS_TEST> 1) {
       printf("(II) NNLS Called with input matrix A:\n");
       for ( l = 0; l < m ; l ++) {
         printf("   ");
         for ( k = 0; k < n ; k ++) printf("%f ", a[k][l]);
         printf("\n");
       }  
       printf(" vector b:\n"); _lss_print_matrix(b,m,1);
    }

    while(1) {

      if (NNLS_TEST> 2)
      {
        printf("(II) New iteration begins..\n");
      }

      /* Find largest positive W[j] */
      for(wmax=0., iz=iz1; iz<=iz2; iz++) {
        j=index[iz]; if(w[j]>wmax) {wmax=w[j]; izmax=iz;}}

      /* Terminate if wmax<=0.; */
      /* it indicates satisfaction of the Kuhn-Tucker conditions */
      if(wmax<=0.0) break;
      iz=izmax; j=index[iz];

      /* The sign of W[j] is ok for j to be moved to set P. */
      /* Begin the transformation and check new diagonal element to avoid */
      /* near linear dependence. */
      asave=a[j][npp1];
      _lss_h12(1, npp1, npp1+1, m, &a[j][0], 1, &up, &dummy, 1, 1, 0);
      unorm=0.;
      if(nsetp!=0) for(l=0; l<nsetp; l++) {d1=a[j][l]; unorm+=d1*d1;}
      unorm=sqrt(unorm);
      d2=unorm+(d1=a[j][npp1], fabs(d1)) * 0.01;
      if((d2-unorm)>0.) {
        /* Col j is sufficiently independent. Copy B into ZZ, update ZZ */
        /* and solve for ztest ( = proposed new value for X[j] ) */
        for(l=0; l<m; l++) zz[l]=b[l];
        _lss_h12(2, npp1, npp1+1, m, &a[j][0], 1, &up, zz, 1, 1, 1);
        ztest=zz[npp1]/a[j][npp1];
        /* See if ztest is positive */
        if(ztest>0.) break;
      }

      /* Reject j as a candidate to be moved from set Z to set P. Restore */
      /* A[npp1,j], set W[j]=0., and loop back to test dual coeffs again */
      a[j][npp1]=asave; w[j]=0.;
    } /* while(1) */
    if(wmax<=0.0) break;

    /* Index j=INDEX[iz] has been selected to be moved from set Z to set P. */
    /* Update B and indices, apply householder transformations to cols in */
    /* new set Z, zero subdiagonal elts in col j, set W[j]=0. */
    for(l=0; l<m; ++l) b[l]=zz[l];
    index[iz]=index[iz1]; index[iz1]=j; iz1++; nsetp=npp1+1; npp1++;
    if(iz1<=iz2) for(jz=iz1; jz<=iz2; jz++) {
      jj=index[jz];
      _lss_h12(2, nsetp-1, npp1, m, &a[j][0], 1, &up, &a[jj][0], 1, m, 1);
    }
    if(nsetp!=m) for(l=npp1; l<m; l++) a[j][l]=0.;
    w[j]=0.;

    if (NNLS_TEST> 2)
    {
       printf("(II) Solving triangular system..\n");
    }
    
    /* Solve the triangular system; store the solution temporarily in Z[] */
    for(l=0; l<nsetp; l++) {
      ip=nsetp-(l+1);
      if(l!=0) for(ii=0; ii<=ip; ii++) zz[ii]-=a[jj][ii]*zz[ip+1];
      jj=index[ip]; zz[ip]/=a[jj][ip];
    }

    if (NNLS_TEST> 2)
    {
       printf("(II) Going secondary loop..\n");
    }

    /* Secondary loop begins here */
    while(++iter<itmax) {
      /* See if all new constrained coeffs are feasible; if not, compute alpha */
      for(alpha=2.0, ip=0; ip<nsetp; ip++) {
        l=index[ip];
        if(zz[ip]<=0.) {t=-x[l]/(zz[ip]-x[l]); if(alpha>t) {alpha=t; jj=ip-1;}}
      }

      /* If all new constrained coeffs are feasible then still alpha==2. */
      /* If so, then exit from the secondary loop to main loop */
      if(alpha==2.0) break;

      if (NNLS_TEST> 2)
      {
        printf("(II) Interpolating..\n");
      }

      /* Use alpha (0.<alpha<1.) to interpolate between old X and new ZZ */
      for(ip=0; ip<nsetp; ip++) {l=index[ip]; x[l]+=alpha*(zz[ip]-x[l]);}

      if (NNLS_TEST> 2)
      {
        printf("(II) Moving coefficients..\n");
      }

      /* Modify A and B and the INDEX arrays to move coefficient i */
      /* from set P to set Z. */
      k=index[jj+1]; pfeas=1;
      do {
        x[k]=0.;
        if(jj!=(nsetp-1)) {
          jj++;
          for(j=jj+1; j<nsetp; j++) {
            ii=index[j]; index[j-1]=ii;
            _lss_g1(a[ii][j-1], a[ii][j], &cc, &ss, &a[ii][j-1]);
            for(a[ii][j]=0., l=0; l<n; l++) if(l!=ii) {
              /* Apply procedure G2 (CC,SS,A(J-1,L),A(J,L)) */
              temp=a[l][j-1];
              a[l][j-1]=cc*temp+ss*a[l][j];
              a[l][j]=-ss*temp+cc*a[l][j];
            }
            /* Apply procedure G2 (CC,SS,B(J-1),B(J)) */
            temp=b[j-1]; b[j-1]=cc*temp+ss*b[j]; b[j]=-ss*temp+cc*b[j];
          }
        }
        npp1=nsetp-1; nsetp--; iz1--; index[iz1]=k;

        if (NNLS_TEST> 2)
        {
          printf("(II) Checking  coefficients..\n");
        }
        /* See if the remaining coeffs in set P are feasible; they should */
        /* be because of the way alpha was determined. If any are */
        /* infeasible it is due to round-off error. Any that are */
        /* nonpositive will be set to zero and moved from set P to set Z */
        for(jj=0, pfeas=1; jj<nsetp; jj++) {
          k=index[jj]; if(x[k]<=0.) {pfeas=0; break;}
        }
      } while(pfeas==0);


      if (NNLS_TEST> 2)
      {
        printf("(II) Copying..\n");
      }

      /* Copy B[] into zz[], then solve again and loop back */
      for(k=0; k<m; k++) zz[k]=b[k];
      for(l=0; l<nsetp; l++) {
        ip=nsetp-(l+1);
        if(l!=0) for(ii=0; ii<=ip; ii++) zz[ii]-=a[jj][ii]*zz[ip+1];
        jj=index[ip]; zz[ip]/=a[jj][ip];
      }
    } /* end of secondary loop */

    if(iter>=itmax) {ret=1; break;}
    for(ip=0; ip<nsetp; ip++) {k=index[ip]; x[k]=zz[ip];}
  } /* end of main loop */
  /* Compute the norm of the final residual vector */
  sm=0.;
  
  if (NNLS_TEST> 2)
  {
    printf("(II) All done..\n");
  }


  if (rnorm != NULL) {
    if (npp1<m) 
      for (k=npp1; k<m; k++) sm+=(b[k]*b[k]);
    else 
      for (j=0; j<n; j++) w[j]=0.;
    *rnorm=sqrt(sm);
  }	
 
  /* Free working space, if it was allocated here */
  if(wp==NULL) free(w); if(zzp==NULL) free(zz); if(indexp==NULL) free(index);
  return(ret);
} /* nnls_ */
/****************************************************************************/

/****************************************************************************/
/** Algorithm for weighting the problem that is given to nnls-algorithm.
    Square roots of weights are used because in nnls the difference
    w*A-w*b is squared.
\return Algorithm returns zero if successful, 1 if arguments are inappropriate.
*/
int nnlsWght(
  /** NNLS dimension N (nr of parameters) */
  int N,
  /** NNLS dimension M (nr of samples) */
  int M,
  /** NNLS matrix A */
  double **A,
  /** NNLS vector B */
  double *b,
  /** Weights for each sample (array of length M) */
  double *weight
) {
  int n, m;
  double *w;

  /* Check the arguments */
  if(N<1 || M<1 || A==NULL || b==NULL || weight==NULL) return(1);

  /* Allocate memory */
  w=(double*)malloc(M*sizeof(double)); if(w==NULL) return(2);

  /* Check that weights are not zero and get the square roots of them to w[] */
  for(m=0; m<M; m++) {
    if(weight[m]<=1.0e-20) w[m]=0.0;
    else w[m]=sqrt(weight[m]);
  }
 
  /* Multiply rows of matrix A and elements of vector b with weights*/
  for(m=0; m<M; m++) {
    for(n=0; n<N; n++) {
      A[n][m]*=w[m];
    }
    b[m]*=w[m];
  }

  free(w);
  return(0);
}
/****************************************************************************/

/****************************************************************************/
/** Algorithm for weighting the problem that is given to nnls-algorithm.
    Square roots of weights are used because in nnls the difference
    w*A-w*b is squared.
    Here user must give squared weights; this makes calculation faster, when
    this function needs to be called many times. 
\return Algorithm returns zero if successful, 1 if arguments are inappropriate.
*/
int nnlsWghtSquared(
  /** NNLS dimension N (nr of parameters) */
  int N,
  /** NNLS dimension M (nr of samples) */
  int M,
  /** NNLS matrix A */
  double **A,
  /** NNLS vector B */
  double *b,
  /** Squared weights for each sample (array of length M) */
  double *sweight
) {
  int n, m;

  /* Check the arguments */
  if(N<1 || M<1 || A==NULL || b==NULL || sweight==NULL) return(1);

  /* Multiply rows of matrix A and elements of vector b with weights*/
  for(m=0; m<M; m++) {
    for(n=0; n<N; n++) {
      A[n][m]*=sweight[m];
    }
    b[m]*=sweight[m];
  }

  return(0);
}
/****************************************************************************/

/****************************************************************************/

