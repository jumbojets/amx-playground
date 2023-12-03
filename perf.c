#include <stdio.h>
#include <stdint.h>
#include <mach/mach_time.h>
#include <arm_neon.h>

#include "amx.h"

#define ITERATIONS 1000000000

#define ITERATE_AMX_OP(op) \
 for (uint64_t i = 0; i < ITERATIONS; ++i) \
  AMX_##op(0LL);

int main() {
  uint64_t start, end;
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(&timebase_info);

  AMX_SET();

  start = mach_absolute_time();

  // for (uint64_t i = 0; i < ITERATIONS; ++i) {
  //   AMX_MAC16(0LL);
  //   AMX_EXTRX(0x201000 >> 10);
  // }
  
  // uint64_t sum = 0; // test latency
  // for (uint64_t i = 0; i < ITERATIONS; ++i) {
  //   // uint_t sum = 0; // test throughput
  //   asm volatile (
  //     "ADD %0, %1, %2\n\t"
  //     : "=r" (sum)
  //     : "r" (sum), "r" (i)
  //   );
  // }

  // test SIMD ADD latency
  // int16_t arr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // int16x8_t a = vld1q_s16(arr);
  // int16x8_t b = vld1q_s16(arr);
  // int16x8_t c;
  // for (uint64_t i = 0; i < ITERATIONS; ++i) {
  //   asm volatile (
  //     "add v0.8h, %1.8h, %2.8h\n\t"
  //     : "=w" (c)
  //     : "w" (a), "w" (b)
  //     : "v0" // clobber list 
  //   );
  // }

  ITERATE_AMX_OP(MAC16);

  for (uint64_t i = 0; i < ITERATIONS; i++) {
    AMX_MAC16(0LL);
    AMX_MAC16(64LL << 20);
  }

  end = mach_absolute_time();

  AMX_CLR();

  uint64_t elapsedNano = (end - start) * timebase_info.numer / timebase_info.denom;

  printf("Total time: %llu nanoseconds\n", elapsedNano);
  printf("Average latency per instruction: %f nanoseconds\n", (double)elapsedNano / ITERATIONS);

  double cpu_speed_GHz = 3.2; // max clock speed on M1 Max 
  printf("Average latency per instruction: %f clock cycles\n", ((double)elapsedNano / ITERATIONS) * cpu_speed_GHz);

  return 0;
}
