#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX
  #define MAX(x,y) ((x)<(y)?(y):(x))
#endif
#ifndef MIN
  #define MIN(x,y) ((x)>(y)?(y):(x))
#endif
#ifndef ABS
  #define ABS(x) MAX(x,-x)
#endif

#if !defined(NORM_IS_MAXVAL) && !defined(NORM_IS_ONENORM)
   #define NORM_IS_ONENORM
#endif
#if !defined(USE_VDPBF16PS) && !defined(DONT_USE_VDPBF16PS)
   #define DONT_USE_VDPBF16PS
#endif

#if !defined(LP64) && !defined(ILP64)
   #define LP64
#endif
#ifdef LP64
   #define myint int
#else
   #define myint size_t
#endif

#if defined(USE_FP16) && defined(USE_BGEMM)
   #error Define USE_BGEMM or USE_FP16 but not both
#endif
#if defined(USE_BGEMM) && defined(USE_SGEMM_NOT_BGEMM)
   #error Define USE_BGEMM or USE_SGEMM_NOT_BGEMM but not both
#endif
#if defined(USE_FP16) && defined(USE_VDPBF16PS)
   #error Define USE_FP16 or USE_VDPBF16PS but not both
#endif

#if defined(USE_FP16) && !defined(USE_SGEMM_NOT_BGEMM) 
   #define USE_SGEMM_NOT_BGEMM
#endif
#if !defined(USE_BGEMM) && !defined(USE_SGEMM_NOT_BGEMM)
   #define USE_SGEMM_NOT_BGEMM
#endif
#ifdef USE_SGEMM_NOT_BGEMM
   #define inttype float
#else
   #define inttype unsigned short
#endif

#ifdef USE_FP16
void rne_convert_fp32_fp16 (const float* in, unsigned short* out, const unsigned int len) {
  unsigned int i, len4;
  float fp32tmp[4];
  unsigned short fp16tmp[8];

  len4 = ((int)(len/4))*4;
  for ( i = 0 ; i < len4 ; i+=4 ) {
     vcvtps2ph_ ( &in[i], &out[i] );
  }
  for ( i = len4 ; i < len ; i++ ) {
     fp32tmp[0] = in[i];
     fp32tmp[1] = in[i];
     fp32tmp[2] = in[i];
     fp32tmp[3] = in[i];
     vcvtps2ph_ ( fp32tmp, fp16tmp );
     out[i] = fp16tmp[0];
  }
}

void convert_fp16_fp32(const unsigned short *in, float* out, unsigned int len)
{
  unsigned int i, len8;
  float fp32tmp[4];
  unsigned short fp16tmp[8];

  len8 = ((int)(len/8))*8;
  for ( i = 0; i < len8 ; i+=8 ) {
     vcvtph2ps_ ( &in[i], &out[i] );
  }
  for ( i = len8 ; i < len ; i++ ) {
     fp16tmp[0] = in[i]; 
     vcvtph2ps_ ( fp16tmp, fp32tmp );
     out[i] = fp32tmp[0];
  }
}
#endif
    
/* we treat bfp16 as unsigned short here */
void rne_convert_fp32_bfp16(const float* in, unsigned short* out, const unsigned int len) {
  unsigned int i = 0;

#ifdef USE_FP16
  rne_convert_fp32_fp16( in, out, len );
  return ;
#endif
  /* truncate buffer to bfp16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;

    memcpy( &int_round, &(in[i]), 4 );

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie even */
    if ( do_round != 0 ) {
      unsigned int fixup = (int_round >> 16) & 1;
      int_round = int_round + 0x00007fff + fixup;
    }

    /* create the bfp16 value by shifting out the lower 16bits */
    int_round = int_round >> 16;

    out[i] = (unsigned short)int_round;
  }
}

union bfp16 {
  float            f;
  unsigned short   i[2];
};

/* we treat bfp16 as unsigned short here */
void convert_bfp16_fp32(const unsigned short* in, float* out, unsigned int len) {
  unsigned int i = 0;

#ifdef USE_FP16
  convert_fp16_fp32( in, out, len );
  return ;
#endif
  /* up-convert is super simple */
  for ( i = 0; i < len; ++i ) {
    union bfp16 t;

    t.i[1] = in[i];
    t.i[0] = 0;
    out[i] = t.f;
  }
}

#ifdef USE_VDPBF16PS
/*
 Does c = c + a * b where a&b are bfloat16, c is float
 Note that vdpbf16ps does 32 products into 16 accumulators
 This duplicates the work across the vectors. This is an a performant code.
 This is just to check accuracy 
*/
void bfloat16_fma_ ( unsigned short *a, unsigned short *b, float *c )
{
   unsigned short at[32];
   unsigned short bt[32];
   float ct[16];
   int i;

   for ( i = 0 ; i < 32 ; i+=2 ) { at[i]= *a; bt[i]= *b; at[i+1]=0; bt[i+1]=0; }
   for ( i = 0 ; i < 16 ; i++ ) { ct[i]= *c; }

/*
   Assume 32 bfloat16 values in a, 32 bfloat16 values in b, 16 fp32 in c
   The instruction vdpbf16ps does: do j = 1, 16
                                      c[j]=a[2*j-1]*b[2*j-1] + a[2*j]*b[2*j]
*/

   vdpbf16ps_ ( at, bt, ct );
   *c = ct[0];
}
#endif

#define A(x,y)     A[((y)-1)*(lda) + ((x)-1)]
#define B(x,y)     B[((y)-1)*(ldb) + ((x)-1)]
#define C(x,y)     C[((y)-1)*(ldc) + ((x)-1)]
#define C0(x,y)    C0[((y)-1)*(ldc) + ((x)-1)]
#define D(x,y)     D[((y)-1)*(ldd) + ((x)-1)]
#define D22(x,y)   D22[((y)-1)*(ldd) + ((x)-1)]
#define D33(x,y)   D33[((y)-1)*(ldd) + ((x)-1)]
#define D44(x,y)   D44[((y)-1)*(ldd) + ((x)-1)]
#define E(x,y)     E[((y)-1)*(lde) + ((x)-1)]
#define sA(x,y)    sA[((y)-1)*(lda) + ((x)-1)]
#define sB(x,y)    sB[((y)-1)*(ldb) + ((x)-1)]
#define sC(x,y)    sC[((y)-1)*(ldc) + ((x)-1)]
#define A1(x,y)    A1[((y)-1)*(lda) + ((x)-1)]
#define B1(x,y)    B1[((y)-1)*(ldb) + ((x)-1)]
#define A2(x,y)    A2[((y)-1)*(lda) + ((x)-1)]
#define B2(x,y)    B2[((y)-1)*(ldb) + ((x)-1)]
#define A3(x,y)    A3[((y)-1)*(lda) + ((x)-1)]
#define B3(x,y)    B3[((y)-1)*(ldb) + ((x)-1)]
#define A4(x,y)    A4[((y)-1)*(lda) + ((x)-1)]
#define B4(x,y)    B4[((y)-1)*(ldb) + ((x)-1)]
#define GARBAGE_VALUE -99.9999

convert_double_to_four_bfloats ( double X, unsigned short *x1, unsigned short *x2, unsigned short *x3, unsigned short *x4 )
{
   float s, stmp;
   double dtmp = X;

   /* Iteration 1: */
   s = (float) dtmp;
   rne_convert_fp32_bfp16 ( &s, x1, 1 );
   convert_bfp16_fp32( x1, &stmp, 1 );
   dtmp -= (double)stmp;

   /* Iteration 2: */
   s = (float) dtmp;
   rne_convert_fp32_bfp16 ( &s, x2, 1 );
   convert_bfp16_fp32( x2, &stmp, 1 );
   dtmp -= (double)stmp;

   /* Iteration 3: */
   s = (float) dtmp;
   rne_convert_fp32_bfp16 ( &s, x3, 1 );
   convert_bfp16_fp32( x3, &stmp, 1 );
   dtmp -= (double)stmp;

   /* Iteration 4: */
   s = (float) dtmp;
   rne_convert_fp32_bfp16 ( &s, x4, 1 );
   convert_bfp16_fp32( x4, &stmp, 1 );
   dtmp -= (double)stmp;

}

void convert_single_to_three_bfloats ( float X, unsigned short *x1, unsigned short *x2, unsigned short *x3 )
{
   float s, stmp;
   double dtmp = (double) X;

   /* Iteration 1: */
   s = (float) dtmp;
   rne_convert_fp32_bfp16 ( &s, x1, 1 );
   convert_bfp16_fp32( x1, &stmp, 1 );
#ifdef PETER_EXPERIMENT
   #include <stdint.h>
   uint32_t *uey = (uint32_t *) &stmp;
   *uey &= ~(1UL << 16); // clear bit 16
   rne_convert_fp32_bfp16 ( &stmp, x1, 1 );
   convert_bfp16_fp32( x1, &stmp, 1 );
#endif
   dtmp -= (double)stmp; 

   /* Iteration 2: */
   s = (float) dtmp;
   rne_convert_fp32_bfp16 ( &s, x2, 1 );
   convert_bfp16_fp32( x2, &stmp, 1 );
   dtmp -= (double)stmp; 

   /* Iteration 3: */
   s = (float) dtmp;
   rne_convert_fp32_bfp16 ( &s, x3, 1 );
   convert_bfp16_fp32( x3, &stmp, 1 );
   dtmp -= (double)stmp; 
}

/* Replaces B with B + A */
void add_matrix_floats_ ( int m, int n, float *A, int lda, float *B, int ldb )
{
   int i, j;

   for ( j = 1 ; j <= n ; j++ ) {
      for ( i = 1; i <= m ; i++ ) {
         B(i,j) += A(i,j);
      }
   }
}

/* Replaces B with B + A */
void add_matrix_float_to_double_ ( int m, int n, float *A, int lda, double *B, int ldb )
{
   int i, j;

   for ( j = 1 ; j <= n ; j++ ) {
      for ( i = 1; i <= m ; i++ ) {
         B(i,j) += (double)A(i,j);
      }
   }
}

void bgemm_nnbeta0 ( int m, int n, int k, float alpha, inttype *A, int lda, inttype *B, int ldb, float *C, int ldc )
{
   int i, j, l;
   float stmp, stmpa, stmpb, zero=0.0;

#ifdef USE_SGEMM_NOT_BGEMM
   char ntrans='N';
  #ifdef ILP64
   size_t m1 = m, n1 = n, k1 = k, lda1 = lda, ldb1 = ldb, ldc1 = ldc;
   sgemm_ ( &ntrans, &ntrans, &m1, &n1, &k1, &alpha, (float *) A, &lda1, (float *) B, &ldb1, &zero, C, &ldc1 );
  #else
   sgemm_ ( &ntrans, &ntrans, &m, &n, &k, &alpha, (float *) A, &lda, (float *) B, &ldb, &zero, C, &ldc );
  #endif 
#else
   for ( j = 1 ; j <= n ; j++ ) {
      for ( i = 1 ; i <= m ; i++ ) {
         stmp = 0.0;
         for ( l = 1; l <= k ; l++ ) {
  #ifdef USE_VDPBF16PS
            bfloat16_fma_ ( &A(i,l), &B(l,j), &stmp );
  #else
            convert_bfp16_fp32( &A(i,l), &stmpa, 1 );
            convert_bfp16_fp32( &B(l,j), &stmpb, 1 );
            stmp += stmpa * stmpb;
  #endif
         }
         if ( alpha == 1.0 ) C(i,j) = stmp ; else C(i,j) = alpha*stmp;
      }
   }
#endif
}

void gregsgemm_ ( char *transa, char *transb, myint *m, myint *n, myint *k, float *alpha, float *A, myint *ldap, float *B, myint *ldbp, float *beta, float *C, myint *ldcp, myint *splitA, myint *splitB, myint *muls, char *addchar )
{
   float stmp, st[3];
   double dtmp;
   inttype *A1, *A2, *A3, *B1, *B2, *B3;
   float *C11, *C12, *C13, *C14, *C21, *C22, *C23, *C24, *C31, *C32, *C33, *C34;
   float *C41, *C42, *C43, *C44;
   unsigned short x[3];
   float *Cd, *Cs;
   double *D;
   myint size1, size2;
   float newalpha = *alpha;
   
   myint i, j, ldd=*m;
   myint lda = *ldap, ldb = *ldbp, ldc = *ldcp;
   myint prods;
   // Here are all our predicates
   int a1=0, a2=0, a3=0, b1=0, b2=0, b3=0;
   int pc11=0, pc12=0, pc13=0, pc21=0, pc22=0, pc23=0, pc31=0, pc32=0, pc33=0;
   int list[9]; // List of matrices needed

   if ( (lda < 1) || (ldb < 1) || (ldc < 1) ) {
      printf("Error. Unsupported leading dimension in gregsgemm()\n");
      exit(-1);
   }
   if ( (*m < 1) || (*n < 1) || (*k < 1) ) {
      printf("Error. Too small parameters for gregsgemm(): %zu %zu %zu\n",(size_t)*m,(size_t)*n,(size_t)*k);
      exit(-1);
   }
   if ( (*splitA < 1) || (*splitA > 3) ) {
      printf("Error. Unsupported split value (%zu) for A in gregsgemm()\n",(size_t)*splitA);
      exit(-1);
   }
   if ( (*splitB < 1) || (*splitB > 3) ) {
      printf("Error. Unsupported split value (%zu) for B in gregsgemm()\n",(size_t)*splitB);
      exit(-1);
   }
   if ( *muls < 1 ) {
      printf("We need at least one multiply in gregsgemm() (muls = %zu)\n",(size_t)*muls);
      exit(-1);
   }
   if ( *muls > (*splitA)*(*splitB) ) {
      printf("Warning. You are requesting more multiples (%zu) than are possible (%zu)\n",(size_t)*muls,(size_t)(*splitA)*(*splitB)); 
      exit(-1);
   }

#ifdef DEBUG
printf("Inside gregsgemm: %c%c mnk=%d %d %d alphabeta=%g %g lda-c=%d %d %d splitA=%d splitB=%d muls=%d addchar=%c\n",*transa,*transb,*m,*n,*k,*alpha,*beta,*ldap,*ldbp,*ldcp,*splitA,*splitB,*muls,*addchar);
#endif

   /* Set up predicates */
   if ( *splitA >= 1 ) a1 = 1; else a1 = 0;
   if ( *splitA >= 2 ) a2 = 1; else a2 = 0;
   if ( *splitA >= 3 ) a3 = 1; else a3 = 0;
   if ( *splitB >= 1 ) b1 = 1; else b1 = 0;
   if ( *splitB >= 2 ) b2 = 1; else b2 = 0;
   if ( *splitB >= 3 ) b3 = 1; else b3 = 0;

   size1 = (*m) * (*k) * sizeof(inttype *);
   size2 = (*k) * (*n) * sizeof(inttype *);
   if ( a1 ) A1=(inttype *) _mm_malloc( size1, 64 );
   if ( a2 ) A2=(inttype *) _mm_malloc( size1, 64 );
   if ( a3 ) A3=(inttype *) _mm_malloc( size1, 64 );
   if ( b1 ) B1=(inttype *) _mm_malloc( size2, 64 );
   if ( b2 ) B2=(inttype *) _mm_malloc( size2, 64 );
   if ( b3 ) B3=(inttype *) _mm_malloc( size2, 64 );

#ifdef USE_FP16
   float scaleA, scaleB, maxabsA, maxabsB;
   //#define MAXFP16 65504
   #define MAXFP16 65000
   maxabsA = A(1,1);
   if ( maxabsA < 0.0 ) maxabsA = -maxabsA;
   for ( j = 1 ; j <= *k; j++ ) {
      for ( i = 1 ; i <= *m; i++ ) {
         if ( *transa=='T' || *transa=='t' ) stmp = A(j,i); else stmp=A(i,j);
         if ( stmp < 0.0 ) stmp = -stmp;
         if ( maxabsA < stmp ) maxabsA = stmp;
      }
   }
   scaleA = MAXFP16 / maxabsA;
   maxabsB = B(1,1);
   if ( maxabsB < 0.0 ) maxabsB = -maxabsB;
   for ( j = 1 ; j <= *n; j++ ) {
      for ( i = 1; i <= *k; i++ ) {
         if ( *transb=='T' || *transb=='t' ) stmp = B(j,i); else stmp=B(i,j);
         if ( stmp < 0.0 ) stmp = -stmp;
         if ( maxabsB < stmp ) maxabsB = stmp;
      }
   }
   scaleB = MAXFP16 / maxabsB;
#endif         

#ifdef USE_FP16
   newalpha = *alpha / (scaleA*scaleB);
#endif

   for ( j = 1 ; j <= *k; j++ ) {
      for ( i = 1; i <= *m; i++ ) {
         if ( *transa=='T' || *transa=='t' ) stmp = A(j,i); else stmp=A(i,j);
#ifdef USE_FP16
         stmp = stmp * scaleA;
#endif
         convert_single_to_three_bfloats ( stmp, &x[0], &x[1], &x[2] );
#ifdef USE_SGEMM_NOT_BGEMM
         convert_bfp16_fp32 ( &x[0], &st[0], 3 );
         if ( a1 ) A1[(j-1)*(*m)+(i-1)] = st[0];
         if ( a2 ) A2[(j-1)*(*m)+(i-1)] = st[1];
         if ( a3 ) A3[(j-1)*(*m)+(i-1)] = st[2];
#else
         if ( a1 ) A1[(j-1)*(*m)+(i-1)] = x[0];
         if ( a2 ) A2[(j-1)*(*m)+(i-1)] = x[1];
         if ( a3 ) A3[(j-1)*(*m)+(i-1)] = x[2];
#endif
      }
   } 
   for ( j = 1 ; j <= *n; j++ ) {
      for ( i = 1; i <= *k; i++ ) {
         if ( *transb=='T' || *transb=='t' ) stmp = B(j,i); else stmp=B(i,j);
#ifdef USE_FP16
         stmp = stmp * scaleB;
#endif
         convert_single_to_three_bfloats ( stmp, &x[0], &x[1], &x[2] );
#ifdef USE_SGEMM_NOT_BGEMM
         convert_bfp16_fp32 ( &x[0], &st[0], 3 );
         if ( b1 ) B1[(j-1)*(*k)+(i-1)] = st[0];
         if ( b2 ) B2[(j-1)*(*k)+(i-1)] = st[1];
         if ( b3 ) B3[(j-1)*(*k)+(i-1)] = st[2];
#else
         if ( b1 ) B1[(j-1)*(*k)+(i-1)] = x[0];
         if ( b2 ) B2[(j-1)*(*k)+(i-1)] = x[1];
         if ( b3 ) B3[(j-1)*(*k)+(i-1)] = x[2];
#endif
      }
   } 

   // Order of operations & matrices: C11,C12,C21,C22,C13,C31,C23,C32,C33
   prods = 0;
   if ( a1 && b1 && (prods < *muls) ) {
       C11=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc11 = 1;
       Cd = C11;
       list[prods++]= 1;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A1, *m, B1, *k, C11, *m );
   }
   if ( a1 && b2 && (prods < *muls) ) {
       C12=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc12 = 1;
       Cd = C12;
       list[prods++]= 2;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A1, *m, B2, *k, C12, *m );
   }
   if ( a2 && b1 && (prods < *muls) ) {
       C21=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc21 = 1;
       Cd = C21;
       list[prods++]= 3;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A2, *m, B1, *k, C21, *m );
   }
   if ( a2 && b2 && (prods < *muls) ) {
       C22=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc22 = 1;
       Cd = C22;
       list[prods++]= 4;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A2, *m, B2, *k, C22, *m );
   }
   if ( a1 && b3 && (prods < *muls) ) {
       C13=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc13 = 1;
       Cd = C13;
       list[prods++]= 5;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A1, *m, B3, *k, C13, *m );
   }
   if ( a3 && b1 && (prods < *muls) ) {
       C31=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc31 = 1;
       Cd = C31;
       list[prods++] = 6;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A3, *m, B1, *k, C31, *m );
   }
   if ( a2 && b3 && (prods < *muls) ) {
       C23=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc23 = 1;
       Cd = C23;
       list[prods++] = 7;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A2, *m, B3, *k, C23, *m );
   }
   if ( a3 && b2 && (prods < *muls) ) {
       C32=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc32 = 1;
       Cd = C32;
       list[prods++] = 8;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A3, *m, B2, *k, C32, *m );
   }
   if ( a3 && b3 && (prods < *muls) ) {
       C33=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc33 = 1;
       Cd = C33;
       list[prods++] = 9;
       bgemm_nnbeta0( *m, *n, *k, newalpha, A3, *m, B3, *k, C33, *m );
   }

   if ( prods != *muls ) {
       printf("Internal gregsgemm() error: Did %zu products but it was requested to do %zu\n",(size_t)prods,(size_t)*muls);
       exit(-1);
   }

#ifdef DEBUG
   printf("Summing %d products in reverse order: ",prods);
   i=0; while ( i < MIN(prods,9) ) { printf("%d ",list[i++]); } printf("\n");
#endif

   // Do sums in reverse order: C11,C12,C21,C22,C13,C31,C23,C32,C33
   if ( (*addchar == 's') || (*addchar == 'S') ) {
      while ( prods >= 2 ) {
         // set Cs according to list[prods-2]
         if      ( list[prods-2] == 1 ) Cs = C11;
         else if ( list[prods-2] == 2 ) Cs = C12;
         else if ( list[prods-2] == 3 ) Cs = C21;
         else if ( list[prods-2] == 4 ) Cs = C22;
         else if ( list[prods-2] == 5 ) Cs = C13;
         else if ( list[prods-2] == 6 ) Cs = C31;
         else if ( list[prods-2] == 7 ) Cs = C23;
         else if ( list[prods-2] == 8 ) Cs = C32;
         else if ( list[prods-2] == 9 ) Cs = C33;
         add_matrix_floats_ ( *m, *n, Cs, *m, Cd, *m );
         --prods;
      }
      for ( j = 1 ; j <= *n ; j++ ) {
         for ( i = 1 ; i <= *m ; i++ ) {
            stmp = (float) Cd[(j-1)*(*m)+(i-1)];
            if ( *beta == 0.0 ) {
               C(i,j) = stmp;
            } else if ( *beta == 1.0 ) {
               C(i,j) = C(i,j) + stmp;
            } else {
               C(i,j) = (*beta)*C(i,j) + stmp;
            }
         }
      }
   } else {
      D = (double *) _mm_malloc((*m)*(*n)*sizeof(double), 64 );
      for ( j = 1 ; j <= *n ; j++ ) {
         for ( i = 1 ; i <= *m ; i++ ) {
            D[(j-1)*(*m)+(i-1)] = (double) Cd[(j-1)*(*m)+(i-1)];
         }
      }
      while ( prods >= 2 ) {
         // set Cs according to list[prods-2]
         if      ( list[prods-2] == 1 ) Cs = C11;
         else if ( list[prods-2] == 2 ) Cs = C12;
         else if ( list[prods-2] == 3 ) Cs = C21;
         else if ( list[prods-2] == 4 ) Cs = C22;
         else if ( list[prods-2] == 5 ) Cs = C13;
         else if ( list[prods-2] == 6 ) Cs = C31;
         else if ( list[prods-2] == 7 ) Cs = C23;
         else if ( list[prods-2] == 8 ) Cs = C32;
         else if ( list[prods-2] == 9 ) Cs = C33;
         add_matrix_float_to_double_( *m, *n, Cs, *m, D, *m );
         --prods;
      }
      for ( j = 1 ; j <= *n ; j++ ) {
         for ( i = 1 ; i <= *m ; i++ ) {
            dtmp = (double) D[(j-1)*(*m)+(i-1)];
            if ( *beta != 0.0 ) dtmp += (*beta)*C(i,j);
            C(i,j) = (float) dtmp;
         }
      }
   }

   if ( (*addchar != 'S') && (*addchar != 's') ) _mm_free(D);
   if ( pc33 ) _mm_free(C33);
   if ( pc32 ) _mm_free(C32);
   if ( pc23 ) _mm_free(C23);
   if ( pc31 ) _mm_free(C31);
   if ( pc13 ) _mm_free(C13);
   if ( pc22 ) _mm_free(C22);
   if ( pc21 ) _mm_free(C21);
   if ( pc12 ) _mm_free(C12);
   if ( pc11 ) _mm_free(C11);
   if ( b3 ) _mm_free(B3);
   if ( b2 ) _mm_free(B2);
   if ( b1 ) _mm_free(B1);
   if ( a3 ) _mm_free(A3);
   if ( a2 ) _mm_free(A2);
   if ( a1 ) _mm_free(A1);
}


// This is just a default SGEMM- same API, just use it the same
void gregdgemm_ ( char *transa, char *transb, myint *m, myint *n, myint *k, double *alpha, double *A, myint *ldap, double *B, myint *ldbp, double *beta, double *C, myint *ldcp, myint *splitA, myint *splitB, myint *muls, char *addchar )
{
   float stmp, st[4];
   double dtmp;
   inttype *A1, *A2, *A3, *A4, *B1, *B2, *B3, *B4;
   float *C11, *C12, *C13, *C14, *C21, *C22, *C23, *C24, *C31, *C32, *C33, *C34;
   float *C41, *C42, *C43, *C44;
   unsigned short x[4];
   float *Cd, *Cs;
   double *D;
   
   myint i, j, ldd=*m;
   myint lda = *ldap, ldb = *ldbp, ldc = *ldcp;
   myint prods;
   // Here are all our predicates
   int a1=0, a2=0, a3=0, a4=0, b1=0, b2=0, b3=0, b4=0;
   int pc11=0, pc12=0, pc13=0, pc21=0, pc22=0, pc23=0, pc31=0, pc32=0, pc33=0;
   int pc14=0, pc24=0, pc34=0, pc41=0, pc42=0, pc43=0, pc44=0;
   int list[16]; // List of matrices needed

   if ( (lda < 1) || (ldb < 1) || (ldc < 1) ) {
      printf("Error. Unsupported leading dimension in gregdgemm()\n");
      exit(-1);
   }
   if ( (*m < 1) || (*n < 1) || (*k < 1) ) {
      printf("Error. Too small parameters for gregdgemm(): %zu %zu %zu\n",(size_t)*m,(size_t)*n,(size_t)*k);
      exit(-1);
   }
   if ( (*splitA < 1) || (*splitA > 4) ) {
      printf("Error. Unsupported split value (%zu) for A in gregdgemm()\n",(size_t)*splitA);
      exit(-1);
   }
   if ( (*splitB < 1) || (*splitB > 4) ) {
      printf("Error. Unsupported split value (%zu) for B in gregdgemm()\n",(size_t)*splitB);
      exit(-1);
   }
   if ( *muls < 1 ) {
      printf("We need at least one multiply in gregdgemm()\n");
      exit(-1);
   }
   if ( *muls > (*splitA)*(*splitB) ) {
      printf("Warning. You are requesting more multiples (%zu) than are possible (%zu)\n",(size_t)*muls,(size_t)(*splitA)*(*splitB)); 
      exit(-1);
   }

#ifdef DEBUG
printf("Inside gregdgemm: %c%c mnk=%d %d %d alphabeta=%g %g lda-c=%d %d %d splitA=%d splitB=%d muls=%d addchar=%c\n",*transa,*transb,*m,*n,*k,*alpha,*beta,*ldap,*ldbp,*ldcp,*splitA,*splitB,*muls,*addchar);
#endif

   /* Set up predicates */
   if ( *splitA >= 1 ) a1 = 1; else a1 = 0;
   if ( *splitA >= 2 ) a2 = 1; else a2 = 0;
   if ( *splitA >= 3 ) a3 = 1; else a3 = 0;
   if ( *splitA >= 4 ) a4 = 1; else a4 = 0;
   if ( *splitB >= 1 ) b1 = 1; else b1 = 0;
   if ( *splitB >= 2 ) b2 = 1; else b2 = 0;
   if ( *splitB >= 3 ) b3 = 1; else b3 = 0;
   if ( *splitB >= 4 ) b4 = 1; else b4 = 0;

   int size1 = (*m) * (*k) * sizeof(inttype);
   int size2 = (*k) * (*n) * sizeof(inttype);
   if ( a1 ) A1=(inttype *) _mm_malloc( size1 , 64);
   if ( a2 ) A2=(inttype *) _mm_malloc( size1 , 64);
   if ( a3 ) A3=(inttype *) _mm_malloc( size1 , 64);
   if ( a4 ) A4=(inttype *) _mm_malloc( size1 , 64);
   if ( b1 ) B1=(inttype *) _mm_malloc( size2 , 64);
   if ( b2 ) B2=(inttype *) _mm_malloc( size2 , 64);
   if ( b3 ) B3=(inttype *) _mm_malloc( size2 , 64);
   if ( b4 ) B4=(inttype *) _mm_malloc( size2 , 64);

   for ( j = 1 ; j <= *k; j++ ) {
      for ( i = 1; i <= *m; i++ ) {
         if ( *transa=='T' || *transa=='t' ) dtmp = A(j,i); else dtmp=A(i,j);
         convert_double_to_four_bfloats ( dtmp, &x[0], &x[1], &x[2], &x[3] );
#ifdef USE_SGEMM_NOT_BGEMM
         convert_bfp16_fp32 ( &x[0], &st[0], 4 );
         if ( a1 ) A1[(j-1)*(*m)+(i-1)] = st[0];
         if ( a2 ) A2[(j-1)*(*m)+(i-1)] = st[1];
         if ( a3 ) A3[(j-1)*(*m)+(i-1)] = st[2];
         if ( a4 ) A4[(j-1)*(*m)+(i-1)] = st[3];
#else
         if ( a1 ) A1[(j-1)*(*m)+(i-1)] = x[0];
         if ( a2 ) A2[(j-1)*(*m)+(i-1)] = x[1];
         if ( a3 ) A3[(j-1)*(*m)+(i-1)] = x[2];
         if ( a4 ) A4[(j-1)*(*m)+(i-1)] = x[3];
#endif
      }
   } 
   for ( j = 1 ; j <= *n; j++ ) {
      for ( i = 1; i <= *k; i++ ) {
         if ( *transb=='T' || *transb=='t' ) dtmp = B(j,i); else dtmp=B(i,j);
         convert_double_to_four_bfloats ( dtmp, &x[0], &x[1], &x[2], &x[3] );
#ifdef USE_SGEMM_NOT_BGEMM
         convert_bfp16_fp32 ( &x[0], &st[0], 4 );
         if ( b1 ) B1[(j-1)*(*k)+(i-1)] = st[0];
         if ( b2 ) B2[(j-1)*(*k)+(i-1)] = st[1];
         if ( b3 ) B3[(j-1)*(*k)+(i-1)] = st[2];
         if ( b4 ) B4[(j-1)*(*k)+(i-1)] = st[3];
#else
         if ( b1 ) B1[(j-1)*(*k)+(i-1)] = x[0];
         if ( b2 ) B2[(j-1)*(*k)+(i-1)] = x[1];
         if ( b3 ) B3[(j-1)*(*k)+(i-1)] = x[2];
         if ( b4 ) B4[(j-1)*(*k)+(i-1)] = x[3];
#endif
      }
   } 

   // Order of ops & matrices: C11,C12,C21,C22,C13,C31,C23,C32,C14,C41,C33,C24,C42,C34,C43,C44
   prods = 0;
   if ( a1 && b1 && (prods < *muls) ) {
       C11=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc11 = 1;
       Cd = C11;
       list[prods++]= 1;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A1, *m, B1, *k, C11, *m );
   }
   if ( a1 && b2 && (prods < *muls) ) {
       C12=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc12 = 1;
       Cd = C12;
       list[prods++]= 2;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A1, *m, B2, *k, C12, *m );
   }
   if ( a2 && b1 && (prods < *muls) ) {
       C21=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc21 = 1;
       Cd = C21;
       list[prods++]= 3;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A2, *m, B1, *k, C21, *m );
   }
   if ( a2 && b2 && (prods < *muls) ) {
       C22=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc22 = 1;
       Cd = C22;
       list[prods++]= 4;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A2, *m, B2, *k, C22, *m );
   }
   if ( a1 && b3 && (prods < *muls) ) {
       C13=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc13 = 1;
       Cd = C13;
       list[prods++]= 5;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A1, *m, B3, *k, C13, *m );
   }
   if ( a3 && b1 && (prods < *muls) ) {
       C31=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc31 = 1;
       Cd = C31;
       list[prods++] = 6;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A3, *m, B1, *k, C31, *m );
   }
   if ( a2 && b3 && (prods < *muls) ) {
       C23=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc23 = 1;
       Cd = C23;
       list[prods++] = 7;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A2, *m, B3, *k, C23, *m );
   }
   if ( a3 && b2 && (prods < *muls) ) {
       C32=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc32 = 1;
       Cd = C32;
       list[prods++] = 8;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A3, *m, B2, *k, C32, *m );
   }
   if ( a1 && b4 && (prods < *muls) ) {
       C14=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc14 = 1;
       Cd = C14;
       list[prods++] = 9;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A1, *m, B4, *k, C14, *m );
   }
   if ( a4 && b1 && (prods < *muls) ) {
       C41=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc41 = 1;
       Cd = C41;
       list[prods++] = 10;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A4, *m, B1, *k, C41, *m );
   }
   if ( a3 && b3 && (prods < *muls) ) {
       C33=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc33 = 1;
       Cd = C33;
       list[prods++] = 11;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A3, *m, B3, *k, C33, *m );
   }
   if ( a2 && b4 && (prods < *muls) ) {
       C24=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc24 = 1;
       Cd = C24;
       list[prods++] = 12;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A2, *m, B4, *k, C24, *m );
   }
   if ( a4 && b2 && (prods < *muls) ) {
       C42=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc42 = 1;
       Cd = C42;
       list[prods++] = 13;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A4, *m, B2, *k, C42, *m );
   }
   if ( a3 && b4 && (prods < *muls) ) {
       C34=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc34 = 1;
       Cd = C34;
       list[prods++] = 14;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A3, *m, B4, *k, C34, *m );
   }
   if ( a4 && b3 && (prods < *muls) ) {
       C43=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc43 = 1;
       Cd = C43;
       list[prods++] = 15;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A4, *m, B3, *k, C43, *m );
   }
   if ( a4 && b4 && (prods < *muls) ) {
       C44=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 ); 
       pc44 = 1;
       Cd = C44;
       list[prods++] = 16;
       bgemm_nnbeta0( *m, *n, *k, *alpha, A4, *m, B4, *k, C44, *m );
   }

   if ( prods != *muls ) {
       printf("Internal gregsgemm() error: Did %zu products but it was requested to do %zu\n",(size_t)prods,(size_t)*muls);
       exit(-1);
   }

#ifdef DEBUG
   printf("Summing %d products in reverse order: ",prods);
   i=0; while ( i < MIN(prods,16) ) { printf("%d ",list[i++]); } printf("\n");
#endif

   // Do sums in reverse order: C11,C12,C21,C22,C13,C31,C23,C32,C33
   if ( (*addchar == 's') || (*addchar == 'S') ) {
      while ( prods >= 2 ) {
         // set Cs according to list[prods-2]
         if      ( list[prods-2] == 1  ) Cs = C11;
         else if ( list[prods-2] == 2  ) Cs = C12;
         else if ( list[prods-2] == 3  ) Cs = C21;
         else if ( list[prods-2] == 4  ) Cs = C22;
         else if ( list[prods-2] == 5  ) Cs = C13;
         else if ( list[prods-2] == 6  ) Cs = C31;
         else if ( list[prods-2] == 7  ) Cs = C23;
         else if ( list[prods-2] == 8  ) Cs = C32;
         else if ( list[prods-2] == 9  ) Cs = C14;
         else if ( list[prods-2] == 10 ) Cs = C41;
         else if ( list[prods-2] == 11 ) Cs = C33;
         else if ( list[prods-2] == 12 ) Cs = C24;
         else if ( list[prods-2] == 13 ) Cs = C42;
         else if ( list[prods-2] == 14 ) Cs = C34;
         else if ( list[prods-2] == 15 ) Cs = C43;
         else if ( list[prods-2] == 16 ) Cs = C44;
         add_matrix_floats_ ( *m, *n, Cs, *m, Cd, *m );
         --prods;
      }
      for ( j = 1 ; j <= *n ; j++ ) {
         for ( i = 1 ; i <= *m ; i++ ) {
            stmp = (float) Cd[(j-1)*(*m)+(i-1)];
            if ( *beta == 0.0 ) {
               C(i,j) = stmp;
            } else if ( *beta == 1.0 ) {
               C(i,j) = C(i,j) + stmp;
            } else {
               C(i,j) = (*beta)*C(i,j) + stmp;
            }
         }
      }
   } else {
      D = (double *) _mm_malloc((*m)*(*n)*sizeof(double), 64 );
      for ( j = 1 ; j <= *n ; j++ ) {
         for ( i = 1 ; i <= *m ; i++ ) {
            D[(j-1)*(*m)+(i-1)] = (double) Cd[(j-1)*(*m)+(i-1)];
         }
      }
      while ( prods >= 2 ) {
         // set Cs according to list[prods-2]
         if      ( list[prods-2] == 1  ) Cs = C11;
         else if ( list[prods-2] == 2  ) Cs = C12;
         else if ( list[prods-2] == 3  ) Cs = C21;
         else if ( list[prods-2] == 4  ) Cs = C22;
         else if ( list[prods-2] == 5  ) Cs = C13;
         else if ( list[prods-2] == 6  ) Cs = C31;
         else if ( list[prods-2] == 7  ) Cs = C23;
         else if ( list[prods-2] == 8  ) Cs = C32;
         else if ( list[prods-2] == 9  ) Cs = C14;
         else if ( list[prods-2] == 10 ) Cs = C41;
         else if ( list[prods-2] == 11 ) Cs = C33;
         else if ( list[prods-2] == 12 ) Cs = C24;
         else if ( list[prods-2] == 13 ) Cs = C42;
         else if ( list[prods-2] == 14 ) Cs = C34;
         else if ( list[prods-2] == 15 ) Cs = C43;
         else if ( list[prods-2] == 16 ) Cs = C44;
         add_matrix_float_to_double_( *m, *n, Cs, *m, D, *m );
         --prods;
      }
      for ( j = 1 ; j <= *n ; j++ ) {
         for ( i = 1 ; i <= *m ; i++ ) {
            dtmp = (double) D[(j-1)*(*m)+(i-1)];
            if ( *beta != 0.0 ) dtmp += (*beta)*C(i,j);
            C(i,j) = (float) dtmp;
         }
      }
   }

   if ( (*addchar != 'S') && (*addchar != 's') ) _mm_free(D);
   if ( pc44 ) _mm_free(C44);
   if ( pc43 ) _mm_free(C43);
   if ( pc34 ) _mm_free(C34);
   if ( pc42 ) _mm_free(C42);
   if ( pc24 ) _mm_free(C24);
   if ( pc33 ) _mm_free(C33);
   if ( pc41 ) _mm_free(C41);
   if ( pc14 ) _mm_free(C14);
   if ( pc32 ) _mm_free(C32);
   if ( pc23 ) _mm_free(C23);
   if ( pc31 ) _mm_free(C31);
   if ( pc13 ) _mm_free(C13);
   if ( pc22 ) _mm_free(C22);
   if ( pc21 ) _mm_free(C21);
   if ( pc12 ) _mm_free(C12);
   if ( pc11 ) _mm_free(C11);
   if ( b4 ) _mm_free(B4);
   if ( b3 ) _mm_free(B3);
   if ( b2 ) _mm_free(B2);
   if ( b1 ) _mm_free(B1);
   if ( a4 ) _mm_free(A4);
   if ( a3 ) _mm_free(A3);
   if ( a2 ) _mm_free(A2);
   if ( a1 ) _mm_free(A1);
}

#ifdef USE_FP16
  int gmh_gemmbits = 22;
  #define MANTISSA 11
#else
  int gmh_gemmbits = 24;
  #define MANTISSA 8
#endif

void gh_set_dgemm_bits (int bits ) {
    if ( bits <= MANTISSA ) gmh_gemmbits = MANTISSA;
    else if ( bits < 2*MANTISSA+1 ) gmh_gemmbits = 2*MANTISSA;
    else if ( bits < 3*MANTISSA+1 ) gmh_gemmbits = 3*MANTISSA;
    else if ( bits < 4*MANTISSA+1 ) gmh_gemmbits = 4*MANTISSA;
    else gmh_gemmbits = 64;
}

// This is just a default SGEMM- same API, just use it the same
void gmhsgemm_ ( char *transa, char *transb, myint *m, myint *n, myint *k, float *alpha, float *A, myint *ldap, float *B, myint *ldbp, float *beta, float *C, myint *ldcp )
{
   myint splitA= 3;     // Split A into 3 parts
   myint splitB= 3;     // Split B into 3 parts
   myint muls = 6;      // Do 6 products in result
   char addchar='S';  // Add result with real*4 only

   if ( gmh_gemmbits >= 8 && gmh_gemmbits <= 64 ) {
      splitA = MAX(MIN(3,gmh_gemmbits/MANTISSA),1);
      splitB = MAX(MIN(3,gmh_gemmbits/MANTISSA),1);
      if ( gmh_gemmbits > 3*MANTISSA ) { muls = 9; addchar='D'; }
      else if ( (gmh_gemmbits <= 3*MANTISSA) && (gmh_gemmbits > 2*MANTISSA) ) muls = 6;
      else if ( (gmh_gemmbits <= 2*MANTISSA) && (gmh_gemmbits > MANTISSA ) ) muls = 3;
      else muls = 1;
   } else {
      printf("Error: default number of bits invalid for gmhsgemm_(): %d (try 8, 16, 24)\n",gmh_gemmbits);
      exit(-1);
   }
   
   gregsgemm_( transa, transb, m, n, k, alpha, A, ldap, B, ldbp, beta, C, ldcp, &splitA, &splitB, &muls, &addchar );
}

// Fortran wrapper for ILP64
void gh_set_dgemm_bits_(long long int * bits)
{
  printf("setting dgemm bits to %lld\n", *bits);
  gh_set_dgemm_bits(*bits);
}

// This is just a default DGEMM- same API, just use it the same
void agemm_ ( char *transa, char *transb, myint *m, myint *n, myint *k, double *alpha, double *A, myint *ldap, double *B, myint *ldbp, double *beta, double *C, myint *ldcp )
{
   myint splitA= 4;     // Split A into 3 parts
   myint splitB= 4;     // Split B into 3 parts
   myint muls = 10;      // Do 6 products in result
   char addchar='D';  // Add result with real*4 only

   if ( gmh_gemmbits >= 8 && gmh_gemmbits <= 64 ) {
      splitA = MAX(MIN(4,gmh_gemmbits/8),1);
      splitB = MAX(MIN(4,gmh_gemmbits/8),1);
      if ( gmh_gemmbits > 32 ) muls = 16; 
      else if ( (gmh_gemmbits <= 32) && (gmh_gemmbits > 24) ) muls = 10;
      else if ( (gmh_gemmbits <= 24) && (gmh_gemmbits > 16) ) muls = 6;
      else if ( (gmh_gemmbits <= 16) && (gmh_gemmbits > 8 ) ) muls = 3;
      else muls = 1;
   } else {
      printf("Error: default number of bits invalid for gmhdgemm_(): %d\n",gmh_gemmbits);
      exit(-1);
   }
   
   if ( gmh_gemmbits <= 32 ) {
       printf("gmh_gemmbits=%d Calling gregdgemm_() with %zux%zu muls=%zu addchar=%c\n",gmh_gemmbits,(size_t)splitA,(size_t)splitB,(size_t)muls,addchar);
       gregdgemm_( transa, transb, m, n, k, alpha, A, ldap, B, ldbp, beta, C, ldcp, &splitA, &splitB, &muls, &addchar );
   } else {
       //printf("gmh_gemmbits=%d Calling Fortran DGEMM\n",gmh_gemmbits);
       dgemm_( transa, transb, m, n, k, alpha, A, ldap, B, ldbp, beta, C, ldcp );
   }
}

/* Sample Usage: 
main( int argc, char **argv )
{
   char transa='N',transb='N', addchar='S';
   int m=8, n=8, k = 8, ldap=8, ldbp=8, ldcp=8;
   int splits=3, muls=6;
   float alpha = 1.0, beta= 0.0;
   float *A, *B, *C; // Fill these in with data
   gregsgemm_ ( &transa, &transb, &m, &n, &k, &alpha, A, &ldap, B, &ldbp, &beta, C, &ldcp, &splits, &muls, &addchar );
}
*/

