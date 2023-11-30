#include <stdio.h>
#include <stdint.h>
#include <mach/mach_time.h>

#include "amx.h"

#define ITERATIONS 10000000000

#define ITERATE_AMX_OP(op) \
 for (uint64_t i = 0; i < ITERATIONS; ++i) \
  AMX_##op(0LL);

int main() {
  uint64_t sum = 0;
  uint64_t start, end;
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(&timebase_info);

  AMX_SET();

  // Record start time
  start = mach_absolute_time();

  for (uint64_t i = 0; i < ITERATIONS; ++i) {
    AMX_MAC16(0LL);
    AMX_EXTRX(0x201000 >> 10);
  }

  // Execute the ADD instruction many times in a loop
  // for (uint64_t i = 0; i < ITERATIONS; ++i) {
  //     asm volatile (
  //         "ADD %0, %1, %2\n\t"
  //         : "=r" (sum)
  //         : "r" (sum), "r" (i)
  //     );
  // }

  // ITERATE_AMX_OP(MAC16);

  // Record end time
  end = mach_absolute_time();

  AMX_CLR();

  // Convert to nanoseconds
  uint64_t elapsedNano = (end - start) * timebase_info.numer / timebase_info.denom;

  printf("Total time: %llu nanoseconds\n", elapsedNano);
  printf("Average latency per instruction: %f nanoseconds\n", (double)elapsedNano / ITERATIONS);

  double cpu_speed_GHz = 3.2; // max clock speed on M1 Max 
  printf("Average latency per instruction: %f clock cycles\n", ((double)elapsedNano / ITERATIONS) * cpu_speed_GHz);

  return 0;
}
