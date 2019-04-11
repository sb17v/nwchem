#include <x86intrin.h>

// Truncate float32->float16, then convert back to float32 and store in original array memory                         
void truncate_sp2hp_(float* s_arr, int* sizeP){
  int size = *sizeP;
  int remainder = size % 8;
  for(int i=0; i<remainder; i++){
    short f16 = _cvtss_sh(s_arr[i], 0);
    s_arr[i]=_cvtsh_ss(f16);
  }
  for(int i=remainder; i<size; i+=8){
    __m256  fp32_vector = _mm256_load_ps(s_arr+i);
    __m128i fp16_vector = _mm256_cvtps_ph(fp32_vector, 0);
    fp32_vector = _mm256_cvtph_ps(fp16_vector);
    _mm256_store_ps(s_arr+i, fp32_vector);
  }
}
