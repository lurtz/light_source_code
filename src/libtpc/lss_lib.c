/*****************************************************************************

 Copyright (c) 2006-2012 by Turku PET Centre

  lss_lib.c 

  This file contains:
    * the routine h12, which constructs and/or applies 
      a single Householder transformation.
    * the routine g1 which computes orthogonal rotation matrix
    * the routine hfti which solves Ax = b
  
  This routine is based on the text and fortran code in
  C.L. Lawson and R.J. Hanson, Solving Least Squares Problems,
  Prentice-Hall, Englewood Cliffs, New Jersey, 1974.


  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 59 Temple
  Place, Suite 330, Boston, MA 02111-1307 USA.

  Turku PET Centre hereby disclaims all copyright interest in the program.
  Juhani Knuuti
  Director, Professor
  Turku PET Centre, Turku, Finland, http://www.turkupetcentre.fi/

  Version history:
  2006       Pauli Sundberg, Kaisa Sederholm
    First created.
  2006-14-07 Pauli Sundberg
    Added _lss_matrix_times_vector( .. )
  2006-01-06 Pauli Sundberg
    Added doxygen documentation style, debug info and hfti()
  2006-24-04 Pauli Sundberg
    Moved h12 function from nnls.c to here since lse.c uses this one too.
  2006-17-05 Pauli Sundberg
    Rewrote some parts of _lss_h12 to be more understoodable. Verified
    _lss_h12, and added debug printing if NDEBUG is not defined
  2007-05-18 VO
    Changed DEBUG to LSS_TEST.
  2012-04-13 VO
    Tiny changes to prevent warnings during compilation.

*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
/****************************************************************************/
#include "include/lss_lib.h"
/****************************************************************************/
                                                            
/****************************************************************************/
/**
* Prints matrix M of m * n sized matrix to stdout
*
* \see _lss_print_matrix2
*/
void  _lss_print_matrix(
  /** matrix to print */
  const double* M,
  /** M dimension is m x n */
  int m, 
  /** M dimension is m x n */
  int n
) {
   int i,j;
   for (i = 0; i < m; i++)
   {
      printf("    ");
      for (j = 0; j < n; j++) printf("%10.10lf ", M[j + i*n]);
      printf("\n");
   }   
}
/****************************************************************************/

/****************************************************************************/
/**
* Prints matrix M of m * n sized matrix to stdout, with increment between
* element being ice and increament between vector icv
*
* \see _lss_print_matrix
*
*/
void  _lss_print_matrix2(
  /** matrix to print */
  const double* M,
  /** M dimension is m x n */
  int m, 
  /** M dimension is m x n */
  int n, 
  /** Increment between elements on matrix */
  int ice, 
  /** Increment between vectors on matrix */
  int icv
) {
   int i,j;
   for (i = 0; i < m; i++)
   {
      printf("    ");
      for (j = 0; j < n; j++) printf("%10.10lf ", M[j*ice + i*icv]);
      printf("\n");
   }   
}
/****************************************************************************/

/****************************************************************************/
/**
* Compares two m * n sized matrix and calculates 
*
* sum{i = 0...m}{ sum{j = 0...n } { (a_i,j - b_i,j)^2 } }
* \return sum of squared element differenses
*/

double _lss_compare_matrix(
 /** Input matrix A size: mxn */ 
 const double* A, 
 /** Input matrix B size: mxn */
 const double* B, 
 /** matrix height */
 int m, 
 /** matrix width */ 
 int n
) {
   int i,j;
   double sum = 0;
   double help;
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
         help = A[j + i*n] - B[j + i*n];
         sum = sum + help*help;
      }
   }
   return sum;	 
}
/****************************************************************************/

/****************************************************************************/
/**
*  Calculates x = M*v, where M is m * n matrix
*
*  actually matrix M is size m_t * n_t, but only first m * n values
*  are used.
*/
void _lss_matrix_times_vector( 
   /** The input matrix, true size m_t * n_t */
   const double* M, 
   /** Dimension used in multiply */
   int m, 
   /** Dimension used in multiply */
   int n, 
   /** The matrix true dimension */
   int m_t, 
   /** The matrix true dimension */
   int n_t, 
   /** Vector that is multiplied with */
   const double* v,          
   /** Increment between element of vector */
   int ie, 
   /** resulting vector */
   double* x
) {
  int i,j;
  double sum = 0;                           
  for (i = 0; i < m ; i ++ ) {
    sum = 0;
    for ( j = 0; j < n; j ++)
      sum += M[i * n_t + j]*v[ ie*j ];
      x[i] = sum;
  }
}
/****************************************************************************/

/****************************************************************************/
/**
 *  Construction and/or application of a single Householder transformation:
 *           Q = I + U*(U**T)/B
 *
 *  Function returns 0 if succesful, or >0 in case of erroneous parameters.
 *
 */
int _lss_h12(
  /** mode=1 to construct and apply a Householder transformation, or
      mode=2 to apply a previously constructed transformation */
  int mode,
  /** Index of the pivot element, on pivot vector */
  int lpivot,
  /** Transformation is constructed to zero elements indexed from l1 to M */
  int l1,
  /** Transformation is constructed to zero elements indexed from l1 to M */
  int m,
  /** With mode=1: On entry, u[] must contain the pivot vector.
     On exit, u[] and up contain quantities defining the vector u[] of
     the Householder transformation.
     With mode=2: On entry, u[] and up should contain quantities previously
     computed with mode=1. These will not be modified. */
  double *u,
  /** u_dim1 is the storage increment between elements. */
  int u_dim1,
  /** with mode=1, here is stored an element defining housholder vector
      scalar, on mode=2 it's only used, and is not modified */
  double *up,
  /** On entry, cm[] must contain the matrix (set of vectors) to which the
     Householder transformation is to be applied. On exit, cm[] will contain
     the set of transformed vectors */
  double *cm,
  /** Storage increment between elements of vectors in cm[] */
  int ice,
  /** Storage increment between vectors in cm[] */
  int icv,
  /** Nr of vectors in cm[] to be transformed;
      if ncv<=0, then no operations will be done on cm[] */
  int ncv
) {
  double d1,  b, clinv, cl, sm;
  int k, j;
  double* w1 = NULL;
  
  if (LSS_TEST > 1)
  {
     printf("(II) LSS_12 called with mode (%d) !\n", mode);
     w1 = (double *)malloc( sizeof(double) * m);
     assert(w1 != NULL);
  }

  /* Check parameters */
  if (mode!=1 && mode!=2) 
     return(1);
  if (m<1 || u==NULL || u_dim1<1 || cm==NULL) 
     assert(0);
//     return(1);
  if (lpivot<0 || lpivot>=l1 || l1>m) 
//     assert(0);
     return(1);

  /* Function Body */
  cl = ABS( u[lpivot*u_dim1] );
  // cl= (d1 = u[lpivot*u_dim1], fabs(d1));

  if (mode==2) 
  { /* Apply transformation I+U*(U**T)/B to cm[] */
    if(cl<=0.) 
//       assert(0);
       return(0);
  } 
  else 
  {   /* Construct the transformation */


     /* This is the way provided in the original pseudocode
     sm = 0;
     for (j = l1; j < m; j++)
     {
        d1 =  u[j * u_dim1];
        sm += d1*d1;
     }
     d1 = u[lpivot * u_dim1];
     sm += d1*d1;
     sm = sqrt(sm);
      
      if (u[lpivot*u_dim1] > 0) 
         sm=-sm;
          
      up[0] = u[lpivot*u_dim1] - sm; 
      u[lpivot*u_dim1]=sm;
      printf("Got sum: %f\n",sm);
     */
  
      /* and this trying to compensate overflow */
      for (j=l1; j<m; j++) 
      {  // Computing MAX 
         cl = MAX( ABS( u[j*u_dim1] ), cl );
      }
      // zero vector?   

      if (cl<=0.) 
         return(0);

       clinv=1.0/cl;
       
       // Computing 2nd power 
       d1=u[lpivot*u_dim1]*clinv; 
       sm=d1*d1;
       
       for (j=l1; j<m; j++) 
       {
          d1=u[j*u_dim1]*clinv; 
          sm+=d1*d1;
       }
       cl *= sqrt(sm); 
       // cl = sqrt( (u[pivot]*clinv)^2 + sigma(i=l1..m)( (u[i]*clinv)^2 ) )

       if (u[lpivot*u_dim1] > 0.) 
          cl=-cl;
       up[0] = u[lpivot*u_dim1] - cl; 
       u[lpivot*u_dim1]=cl;
  }

  // no vectors where to apply? only change pivot vector!	

  b=up[0] * u[lpivot*u_dim1];
  
  /* b must be nonpositive here; if b>=0., then return */
  if (b == 0 ) 
//     assert(0);
     return(0);
     
  
  // ok, for all vectors we want to apply
  for (j =0; j < ncv; j++) 
  {
    // take s = c[p,j]*h + sigma(i=l..m){ c[i,j] *v [ i ] }
    // 
    sm = cm[ lpivot * ice + j * icv ] * (up[0]);
    for (k=l1; k<m; k++) 
    {
       sm += cm[ k * ice + j*icv ] * u[ k*u_dim1 ]; 
    }
    
    if (sm != 0.0) 
    {
      sm *= (1/b); 
      // cm[lpivot, j] = ..
      cm[ lpivot * ice + j*icv] += sm * (up[0]);
      // printf("        P applying to: (%d,%d) = %f \n",lpivot,j,cm[ lpivot * ice + j*icv]);
      // for i = l1...m , set c[i,j] = c[i,j] + s*v[i]
      for (k= l1; k<m; k++) 
      {
         cm[ k*ice + j*icv] += u[k * u_dim1]*sm;
         // printf("        applying to: (%d,%d) = %f\n",k,j, cm[ k*ice + j*icv]);
      }
    }
  }
  
   if (LSS_TEST > 1)
   {
     // just to make things pretty for DEBUG, create the matrix
     
     printf("Housholder transformation matrix:\n");
     for (k= 0; k < lpivot; k++ )
        w1[k] = 0;
     w1[lpivot] = up[ 0 ];
     for (k= lpivot + 1; k < l1; k++ )
        w1[k] = 0;
     for (k= l1; k < m; k++ )
        w1[k]  = u[ k*u_dim1];
        
     // ok we now have the u vector of the
     // Q = I + b^-1 * u*u^T
     // construct Q
     for( k = 0; k < m; k++)
     {
        for( j = 0; j < m; j ++)
        {
           d1 = w1[k]*w1[j];
           d1 = d1 * 1/b;
           
           if (k == j)
           {
              d1 = d1 + 1;
           }
           printf("%10.10lf ", d1 );
        }
        printf("\n");
     }
  }   
  
  return(0);
} /* lss_h12 */
/****************************************************************************/

/****************************************************************************/
/**
 *  Compute orthogonal rotation matrix:
 *    (C, S) so that (C, S)(A) = (sqrt(A**2+B**2))
 *    (-S,C)         (-S,C)(B)   (   0          )
 *  Compute sig = sqrt(A**2+B**2):
 *    sig is computed last to allow for the possibility that sig may be in
 *    the same location as A or B.
 */
void _lss_g1(double a, double b, double *cterm, double *sterm, double *sig)
{
  double d1, xr, yr;

  if(fabs(a)>fabs(b)) {
    xr=b/a; d1=xr; yr=sqrt(d1*d1 + 1.); d1=1./yr;
    *cterm=(a>=0.0 ? fabs(d1) : -fabs(d1));
    *sterm=(*cterm)*xr; *sig=fabs(a)*yr;
  } else if(b!=0.) {
    xr=a/b; d1=xr; yr=sqrt(d1*d1 + 1.); d1=1./yr;
    *sterm=(b>=0.0 ? fabs(d1) : -fabs(d1));
    *cterm=(*sterm)*xr; *sig=fabs(b)*yr;
  } else {
    *sig=0.; *cterm=0.; *sterm=1.;
  }
} /* _lss_g1 */
/****************************************************************************/

/****************************************************************************/
/**
 *  Solve least squares problem by housholder transformations 
 *
 *  Solves Ax = b, but B can be matrix when it's columns are considered as
 *  distinct solutions
 * 
 *  On successfull exit matrix B will contain the corresponding solution 
 *  vectors
 *
   \return 
      0 on success
      1 on failure ( dimension error, out of memory )
 */
int _lss_hfti(
   /** A is a m_a * n_a matrix, defining the A'x = b, where A' is first m 
       columns of A */
  double* A, 

  /** matrix A dimension (true size) */
  int m_a, 
  /** matrix A dimension, dimension that will be user to solve the eq */
  int m,
  /** matrix A dimension (true size) */
  int n_a, 
  /** matrix A dimension, dimension that will be used to solve the eq */
  int n, 

  /** B is a m_b * n_b size matrix defining the right hand side vectors b_i */
  double* B, 
  /** matrix B dimension */
  int m_b, 
  /** matrix B dimension */
  int n_b, 
  /** smallest number that is considered to be non-zero */
  double tau,
   /** will contain pseudorank of the matrix on succesfull calculation */
   int* krank, 
   /** vector of size n_b * 1, will containt residual norm of the solutin after successfull 
       calculation, can be NULL */
   double* rnorm, 
   /** working space 1, n_a * 1 sized vector (array), 
       if NULL then it will be allocated (and deleted) by the routine */
   double* w1, 
   /** working space 2, n_a * 1 sized vector (array), 
       if NULL then it will be allocated (and deleted) by the routine */
   double* w2, 
   /** n_a * 1 vector where routine records indices describing 
       the permutation of column vectors, can be NULL */
   int* ip
) {
   double help = 0;
   double sm    = 0;
   // double factor = 0.001;
   int loop = 0;
   int ldiag = MIN(m, n);
   int j = 0;
   int i = 0;
   int lmax = 0;
   int bloop = 0;

   char w1_alloc = 0;
   char w2_alloc = 0;
   char w3_alloc = 0;

   if (w1 == NULL)
   {
      if (LSS_TEST > 2)
      {
         printf("allocating for w1!\n");	
      }
      w1 = (double *)malloc( sizeof(double) * n );
       w1_alloc = 1;
   }
   if (w2 == NULL)
   {
      if (LSS_TEST > 2)
      {
        printf("allocating for w2!\n");	
      }
      w2 = (double *)malloc ( sizeof(double) * n);
      w2_alloc = 1;
   }

   if (ip == NULL)
   {
      if (LSS_TEST > 2)
      {
         printf("allocating for ip!\n");	
      }
      ip = (int *)malloc ( sizeof(int) * n);
      w3_alloc = 1;
   }
   if ( w1 == NULL || w2 == NULL || ip == NULL)
   {
      return 1;
   }

   for ( loop = 0; loop < n; loop ++)
   {
      ip[loop] = 0;
      w1[loop] = 0;
      w2[loop] = 0;
   }
   
   if (LSS_TEST > 0)
      printf("(WW) LSS_HFTI called with m=%d, m_a=%d, n=%d, n_a=%d\n", m, m_a, n, n_a);
   assert( ldiag > 0 );
   
   assert( m_a >= m );
   assert( n_a >= n );
//   assert( n >= m);
   
   /* note: this would be faster if we would do this other way, aka
      not warping on array but going in order */
   if (LSS_TEST > 1)
   {
      printf("Ldiag solved as: %d\n", ldiag);
      printf("matrix A:\n");
      _lss_print_matrix2(A, m, n, 1 , n_a);
      printf("matrix B:\n");
      _lss_print_matrix(B,m_b, n_b);
   }

   for (j = 0; j < ldiag; j++)
   {
      /*  Update squared column lengths and find lmax
      if (j > 0)
      {
         lmax = j;
         for ( loop = j; loop < n_a; loop ++)
         {
            w1[loop] = w1[loop] -  A[loop + (j-1)* n_a] * A[loop + (j-1)*n_a];
            if (w1[loop] > w1[lmax])
            {
               lmax = loop;
            }
         }
      }
      fixme: we assume that the precission is enough */

      lmax = j;
      for ( loop = j; loop < n; loop++)
      {
         w1[loop] = 0;  
         for ( i = j; i < m; i++)
         {
            w1[loop] = w1[loop] + A[loop+ i*n_a]* A[loop+i*n_a];
         }
         if (w1[loop] > w1[lmax])
         {
            lmax = loop;
         }
      }
      /* lmax is now ok. Do interchanges if needed */
      ip[j] = lmax;
      if (ip[j] != j)
      { 
         // interchange columns lmax and j
         for (i = 0; i < m; i ++)
         {
            // swap A[i,j] and A[i,lmax]
            help = A[j + i* n_a];
            A[j + i*n_a]    = A[lmax + i*n_a];
            A[lmax +i*n_a] = help;
         }
         w1[lmax] = w1[j];
         // note: now w1[j] is incorrect
      }

      // compute the j-th transformation and apply it to A and B
      
      // assert( _lss_h12(1,j,j+1,m,&(A[j]), n_a, &(w1[j]), &(A[(j+1)]), n_a, 1, n_a - j - 1) == 0);
      // assert( _lss_h12(2,j,j+1,m,&(A[j]), n_a, &(w1[j]), &(B[0])    , n_b, 1, n_b    ) == 0);

      // we want to access columns
      _lss_h12(1,j,j+1,m,&(A[j]), n_a, &(w1[j]), &(A[(j+1)]), n_a, 1, n - j - 1);
      _lss_h12(2,j,j+1,m,&(A[j]), n_a, &(w1[j]), &(B[0])    , n_b, 1, n_b    );
      
      if (LSS_TEST > 1) {
         printf("converting..\n");
         printf("%d matrix A:\n", j); _lss_print_matrix2(A,m, n, 1 , n_a);
         printf("%d matrix B:\n", j); _lss_print_matrix(B,m_b, n_b);
      }
   }
   
   // ok determine pseudorank
   (*krank) = 0;
   while( ABS( A[ (*krank) + (*krank)*n_a ] ) >= tau ) {
      (*krank) ++;
      // check boundaries
      if ((*krank) >= ldiag) break;
   }
   // now A[k,k] < tau
   (*krank) --;
      
   // now krank is the last index with A[k,k] >= tau
     
   // compute norms of the residual vector   
   // if the param was != NULL
   if (rnorm != NULL) {
      printf("(WW) Rnorm is not implimented!\n");
   }
  
   if (LSS_TEST > 1)  
      printf("Pseudorank= %d, tau=%f, ldiag=%d\n", *krank,tau,ldiag);
   
   // If the pseudorank (*krank) is less than n compute housholder
   // decomposition of first k rows
      
   assert( (*krank) < n);
      
   if (*krank < n - 1) {
     for ( loop = *krank; loop >= 0; loop --) {
       // pivot is A[i,loop]
       // access rows
       _lss_h12(1, loop, *krank+1, n, &A[ loop*n_a + 0], 1 , &w2[loop], &A[0], 1, n_a, loop );
       if (LSS_TEST > 2) {
         printf("row loop=%d, matrix A:\n", loop); _lss_print_matrix2(A,m,n,1,n_a);
         printf("row loop=%d, matrix B:\n", loop); _lss_print_matrix(B,m_b,n_b);
       }
     }
   } else { }

   // solve for all B vectors
   for ( bloop = 0; bloop < n_b ; bloop ++)
   {
      // solve the k*k triangular system
      // calculate the result element i,j
      
      // solve the first, krank <= m_b
      // we have a_k,k * x_k = b_k
      B[(*krank)*n_b + bloop ] = ( B[(*krank)*n_b + bloop] ) / A[ (*krank) + (*krank) * n_a ] ;
      
      // and then the rest   
      for (i = *krank - 1; i >= 0; i --)
      {
         sm = 0;
         for (j = i + 1; j <= (*krank); j++)
            sm = sm + A[i*n_a + j] * B[j*n_b  + bloop]; 
         B[i * n_b + bloop] = ( B[i * n_b + bloop ] - sm ) / A[i + i * n_a ] ;
      }
 
      // ok we have the transformed solution, now transform it to be a solution
      // for the original problem
      
      // zero the unsolved 
      for (j = *krank + 1; j < n ; j++)
         B[bloop + j*n_b] = 0;

      for (i=0;i <= *krank; i++)
      {
        // access rows
        assert( _lss_h12(2,i, *krank +1, n, &A[i*n_a + 0], 1, &w2[i], &B[ 0*n_b + bloop ], n_b, 1, 1) == 0);
      }
      
      // and we did some interchanges to columns?
      for (j = ldiag - 1; j >= 0; j--)
      {
         if ( ip[j] != j )
         {
            help = B[j*n_b +  bloop];
            B[ j * n_b + bloop] = B[ ip[j]*n_b + bloop];
            B[ ip[j]*n_b + bloop] = help;
         }
      }
   } // end of for all B vectors
     
   if (LSS_TEST > 1) {
     printf("matrix A:\n"); _lss_print_matrix2(A,m, n, 1 , n_a);
     printf("matrix X:\n"); _lss_print_matrix(B,m_b, n_b);
   }

   // ok we are done! solution is now on B matrix
   if (LSS_TEST > 0) printf("(II) HFTI DONE!\n");

   if (w1_alloc) {free(w1);}
   if (w2_alloc) {free(w2);}
   if (w3_alloc) {free(ip);}

   return 0;
}
/****************************************************************************/

/****************************************************************************/
