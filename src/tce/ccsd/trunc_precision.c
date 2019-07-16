#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __AVX2__
#include <x86intrin.h>
#endif

// Truncate float32->float16, then convert back to float32 and store in original array memory
void truncate_sp2hp_(float* s_arr, int* sizeP)
{
  const int size = *sizeP;

  // CVTP[SH] instructions were introduced in Ivy Bridge,
  // but we test for Haswell because it's easier.
#ifdef __AVX2__
  int i=0;
  for(; i<size%8; i+=8) {
    __m256  fp32_vector = _mm256_load_ps(s_arr+i);
    __m128i fp16_vector = _mm256_cvtps_ph(fp32_vector, 0);
    fp32_vector = _mm256_cvtph_ps(fp16_vector);
    _mm256_store_ps(s_arr+i, fp32_vector);
  }
  for(; i<size; i++) {
    int16_t f16 = _cvtss_sh(s_arr[i], 0);
    s_arr[i]=_cvtsh_ss(f16);
  }
#else
#error No support for float16 conversion.
// We can use https://github.com/Maratyszcza/FP16 for this...
#endif
}

void truncate_sp2bf_(float* s_arr, int* sizeP)
{
  const int size = *sizeP;
  for (int i=0; i<size; i++) {
#if 0
    // shift is the right way to convert to/from BF16...
    uint32_t tmp = (*(uint32_t*)&s_arr[i] >> 16) << 16;
    s_arr[i] = *(float*)&tmp;
#else
    uint32_t tmp = *(uint32_t*)&s_arr[i];
    tmp &= ~0xFFFF;
    s_arr[i] = *(float*)&tmp;
#endif
  }
}

#if !defined(LP64) && !defined(ILP64)
   #define LP64
#endif
#ifdef LP64
   #define myint int
#else
   #define myint size_t
#endif

#define MAXFP16 65000  /* Maximum FP16 value is actually larger, but... */

/* inttype is the actual physical storage for the triplets of matrices like A becomes A1, A2, A3. If strictly using a bfloat16 type or fp16 type, change this to unsigned short, but if simulating bfloat16/fp16 via sgemm() calls, leave this as "float" format so sgemm() references work. In practice, the sgemm() references should be replaced with bf16/fp16-gemm based codes, and inttype should be changed to unsigned short. */
#define inttype float

void rne_convert_fp32_fp16 (const float* in, unsigned short* out, const unsigned int len) {

  /* Using similar code as above, but I need a destination not just a source */
  /* No idea if this actually works */
#ifdef __AVX2__
  unsigned int i, len4;
  __m128  fp32tmp;
  __m128i fp16tmp;
  len4 = ((int)(len/4))*4;
  for ( i = 0 ; i < len4 ; i+=4 ) {
     fp32tmp = _mm_load_ps ( &in[i] );
     fp16tmp = _mm_cvtps_ph ( fp32tmp, 0 );
     _mm_store_ps( &out[i], fp16tmp );
  }
  /* Todo: Use masks instead of the below, but doesn't that only work for AVX-512? */
  for ( i = len4 ; i < len ; i++ ) {
     fp32tmp[0] = in[i];
     fp32tmp[1] = in[i];
     fp32tmp[2] = in[i];
     fp32tmp[3] = in[i];
     fp16tmp = _mm_cvtps_ph ( fp32tmp, 0 );
     out[i] = fp16tmp[0];
  }
#else
#error No support for float16 conversion.
// We can use https://github.com/Maratyszcza/FP16 for this...
#endif

}

void convert_fp16_fp32(const unsigned short *in, float* out, unsigned int len)
{
  unsigned int i, len8;
  __m128  fp32tmp;
  __m128i fp16tmp;

  len8 = ((int)(len/8))*8;
  for ( i = 0; i < len8 ; i+=8 ) {
     fp16tmp = _mm_load_ps ( &in[i] );
     fp32tmp = _mm_cvtph_ps ( fp16tmp );
     _mm_store_ps ( &out[i], fp32tmp );
  }
  for ( i = len8 ; i < len ; i++ ) {
     fp16tmp[0] = in[i];
     fp32tmp = _mm_cvtph_ps ( fp16tmp );
     out[i] = fp32tmp[0];
  }
}

/* we treat bfp16 as unsigned short here */
void rne_convert_fp32_bfp16(const float* in, unsigned short* out, const unsigned int len) {
  unsigned int i = 0;

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

  /* up-convert is super simple */
  for ( i = 0; i < len; ++i ) {
    union bfp16 t;

    t.i[1] = in[i];
    t.i[0] = 0;
    out[i] = t.f;
  }
}

#define A(x,y)     A[((y)-1)*(lda) + ((x)-1)]
#define B(x,y)     B[((y)-1)*(ldb) + ((x)-1)]
#define C(x,y)     C[((y)-1)*(ldc) + ((x)-1)]
#define A1(x,y)    A1[((y)-1)*(lda) + ((x)-1)]
#define B1(x,y)    B1[((y)-1)*(ldb) + ((x)-1)]
#define A2(x,y)    A2[((y)-1)*(lda) + ((x)-1)]
#define B2(x,y)    B2[((y)-1)*(ldb) + ((x)-1)]
#define A3(x,y)    A3[((y)-1)*(lda) + ((x)-1)]
#define B3(x,y)    B3[((y)-1)*(ldb) + ((x)-1)]

void convert_single_to_three_smaller_units ( float X, unsigned short *x1, unsigned short *x2, unsigned short *x3, char *modetype )
{
   float s, stmp;
   double dtmp = (double) X;

   if ( *modetype == 'F' ) { /* FP16 */
      /* Iteration 1: */
      s = (float) dtmp;
      rne_convert_fp32_fp16 ( &s, x1, 1 );
      convert_fp16_fp32( x1, &stmp, 1 );
      dtmp -= (double)stmp;

      /* Iteration 2: */
      s = (float) dtmp;
      rne_convert_fp32_fp16 ( &s, x2, 1 );
      convert_fp16_fp32( x2, &stmp, 1 );
      dtmp -= (double)stmp;

      /* Iteration 3: */
      s = (float) dtmp;
      rne_convert_fp32_fp16 ( &s, x3, 1 );
      convert_fp16_fp32( x3, &stmp, 1 );
   } else if ( *modetype == 'B' ) { /* BFP16 */
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
   } else {
      printf("Unknown modetype in convert_single_to_three_smaller_units()=%c\n",*modetype);
      exit(-1);
   }
}

/* Replaces B with B + A using the macro defns for A and B above */
void add_matrix_floats_ ( int m, int n, float *A, int lda, float *B, int ldb )
{
   int i, j;

   for ( j = 1 ; j <= n ; j++ ) {
      for ( i = 1; i <= m ; i++ ) {
         B(i,j) += A(i,j);
      }
   }
}

void sgemmreplacement_ ( char *transa, char *transb, myint *m, myint *n, myint *k, float *alpha, float *A, myint *ldap, float *B, myint *ldbp, float *beta, float *C, myint *ldcp, myint *splitA, myint *splitB, myint *muls, char *modetype )
{
   float stmp, st[3];
   double dtmp;
   inttype *A1, *A2, *A3, *B1, *B2, *B3;
   float *C11, *C12, *C13, *C14, *C21, *C22, *C23, *C24, *C31, *C32, *C33, *C34;
   float *C41, *C42, *C43, *C44;
   unsigned short x[3];
   float *Cd, *Cs;
   myint size1, size2;
   float newalpha = *alpha;
   const double dzero = 0.0;
   char ntrans='N';
   float scaleA, scaleB, maxabsA, maxabsB;
   myint i, j, ldd=*m;
   myint lda = *ldap, ldb = *ldbp, ldc = *ldcp;
   myint prods;
   // Here are all our predicates
   int a1=0, a2=0, a3=0, b1=0, b2=0, b3=0;
   int pc11=0, pc12=0, pc13=0, pc21=0, pc22=0, pc23=0, pc31=0, pc32=0, pc33=0;
   int list[9]; // List of matrices needed

   if ( (lda < 1) || (ldb < 1) || (ldc < 1) ) {
      printf("Error. Unsupported leading dimension in sgemmreplacement()\n");
      exit(-1);
   }
   if ( (*m < 1) || (*n < 1) || (*k < 1) ) {
      printf("Error. Too small parameters for sgemmreplacement(): %zu %zu %zu\n",(size_t)*m,(size_t)*n,(size_t)*k);
      exit(-1);
   }
   if ( (*splitA < 1) || (*splitA > 3) ) {
      printf("Error. Unsupported split value (%zu) for A in sgemmreplacement()\n",(size_t)*splitA);
      exit(-1);
   }
   if ( (*splitB < 1) || (*splitB > 3) ) {
      printf("Error. Unsupported split value (%zu) for B in sgemmreplacement()\n",(size_t)*splitB);
      exit(-1);
   }
   if ( *muls < 1 ) {
      printf("We need at least one multiply in sgemmreplacement() (muls = %zu)\n",(size_t)*muls);
      exit(-1);
   }
   if ( *muls > (*splitA)*(*splitB) ) {
      printf("Warning. You are requesting more multiples (%zu) than are possible (%zu)\n",(size_t)*muls,(size_t)(*splitA)*(*splitB)); 
      exit(-1);
   }

#ifdef DEBUG
printf("Inside sgemmreplacement: %c%c mnk=%d %d %d alphabeta=%g %g lda-c=%d %d %d splitA=%d splitB=%d muls=%d\n",*transa,*transb,*m,*n,*k,*alpha,*beta,*ldap,*ldbp,*ldcp,*splitA,*splitB,*muls);
#endif

   /* Set up predicates */
   if ( *splitA >= 1 ) a1 = 1; else a1 = 0;
   if ( *splitA >= 2 ) a2 = 1; else a2 = 0;
   if ( *splitA >= 3 ) a3 = 1; else a3 = 0;
   if ( *splitB >= 1 ) b1 = 1; else b1 = 0;
   if ( *splitB >= 2 ) b2 = 1; else b2 = 0;
   if ( *splitB >= 3 ) b3 = 1; else b3 = 0;

   size1 = (*m) * (*k) * sizeof(inttype);
   size2 = (*k) * (*n) * sizeof(inttype);
   /* Make certain inttype is specified correctly: see above, should be float if we're using SGEMM() to simulate things, otherwise it should be unsigned short */
   if ( a1 ) A1=(inttype *) _mm_malloc( size1, 64 );
   if ( a2 ) A2=(inttype *) _mm_malloc( size1, 64 );
   if ( a3 ) A3=(inttype *) _mm_malloc( size1, 64 );
   if ( b1 ) B1=(inttype *) _mm_malloc( size2, 64 );
   if ( b2 ) B2=(inttype *) _mm_malloc( size2, 64 );
   if ( b3 ) B3=(inttype *) _mm_malloc( size2, 64 );

   /* Special scaling is required when FP16 is used: make sure all numbers are in range */
   if ( *modetype == 'F' ) {
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
      newalpha = *alpha / (scaleA*scaleB);
   }

   for ( j = 1 ; j <= *k; j++ ) {
      for ( i = 1; i <= *m; i++ ) {
         if ( *transa=='T' || *transa=='t' ) stmp = A(j,i); else stmp=A(i,j);
         if ( *modetype == 'F' ) {
            stmp *= scaleA;
            convert_single_to_three_smaller_units ( stmp, &x[0], &x[1], &x[2], modetype );
            convert_fp16_fp32 ( &x[0], &st[0], 3 );
         } else {
            convert_single_to_three_smaller_units ( stmp, &x[0], &x[1], &x[2], modetype );
            convert_bfp16_fp32 ( &x[0], &st[0], 3 );
         }
         if ( a1 ) A1[(j-1)*(*m)+(i-1)] = st[0];
         if ( a2 ) A2[(j-1)*(*m)+(i-1)] = st[1];
         if ( a3 ) A3[(j-1)*(*m)+(i-1)] = st[2];
      }
   }
   for ( j = 1 ; j <= *n; j++ ) {
      for ( i = 1; i <= *k; i++ ) {
         if ( *transb=='T' || *transb=='t' ) stmp = B(j,i); else stmp=B(i,j);
         if ( *modetype == 'F' ) {
            stmp *= scaleA;
            convert_single_to_three_smaller_units ( stmp, &x[0], &x[1], &x[2], modetype );
            convert_fp16_fp32 ( &x[0], &st[0], 3 );
         } else {
            convert_single_to_three_smaller_units ( stmp, &x[0], &x[1], &x[2], modetype );
            convert_bfp16_fp32 ( &x[0], &st[0], 3 );
         }
         if ( b1 ) B1[(j-1)*(*k)+(i-1)] = st[0];
         if ( b2 ) B2[(j-1)*(*k)+(i-1)] = st[1];
         if ( b3 ) B3[(j-1)*(*k)+(i-1)] = st[2];
      }
   }

   /* In this code we malloc a temporary array for EVERY multiply. Obviously,
    * that is overkill and unnecessary. We could reuse them, but that would
    * also imply that we reform the matrix as we go instead of one step at the
    * end. Instead, this code is setup to run every GEMM independently and at
    * the same time on its own array. ONLY at the end, the data is reformed. */
   // Order of operations & matrices: C11,C12,C21,C22,C13,C31,C23,C32,C33
   prods = 0;
   if ( a1 && b1 && (prods < *muls) ) {
       C11=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc11 = 1;
       Cd = C11;
       list[prods++]= 1;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A1, m, (float *) B1, k, &dzero, Cd, m );
   }
   if ( a1 && b2 && (prods < *muls) ) {
       C12=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc12 = 1;
       Cd = C12;
       list[prods++]= 2;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A1, m, (float *) B2, k, &dzero, Cd, m );
   }
   if ( a2 && b1 && (prods < *muls) ) {
       C21=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc21 = 1;
       Cd = C21;
       list[prods++]= 3;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A2, m, (float *) B1, k, &dzero, Cd, m );
   }
   if ( a2 && b2 && (prods < *muls) ) {
       C22=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc22 = 1;
       Cd = C22;
       list[prods++]= 4;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A2, m, (float *) B2, k, &dzero, Cd, m );
   }
   if ( a1 && b3 && (prods < *muls) ) {
       C13=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc13 = 1;
       Cd = C13;
       list[prods++]= 5;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A1, m, (float *) B3, k, &dzero, Cd, m );
   }
   if ( a3 && b1 && (prods < *muls) ) {
       C31=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc31 = 1;
       Cd = C31;
       list[prods++] = 6;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A3, m, (float *) B1, k, &dzero, Cd, m );
   }
   if ( a2 && b3 && (prods < *muls) ) {
       C23=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc23 = 1;
       Cd = C23;
       list[prods++] = 7;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A2, m, (float *) B3, k, &dzero, Cd, m );
   }
   if ( a3 && b2 && (prods < *muls) ) {
       C32=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc32 = 1;
       Cd = C32;
       list[prods++] = 8;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A3, m, (float *) B2, k, &dzero, Cd, m );
   }
   if ( a3 && b3 && (prods < *muls) ) {
       C33=(float *) _mm_malloc((*m)*(*n)*sizeof(float), 64 );
       pc33 = 1;
       Cd = C33;
       list[prods++] = 9;
       sgemm_ ( &ntrans, &ntrans, m, n, k, &newalpha, (float *)A3, m, (float *) B3, k, &dzero, Cd, m );
   }

   if ( prods != *muls ) {
       printf("Internal sgemmreplacement() error: Did %zu products but it was requested to do %zu\n",(size_t)prods,(size_t)*muls);
       exit(-1);
   }

   /* Note that we can save memory usage by summing into the destination array
    * as we go. */
   // Do sums in reverse order: C11,C12,C21,C22,C13,C31,C23,C32,C33
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
