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
    s_arr[i] &= ~0xFFFF;
#endif
  }
}
