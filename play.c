#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "amx.h"
#include "util.h"

// mini matrix multiplication
#define N 8

__fp16 A[N*N]; // assume that it is transposed
__fp16 B[N*N];
__fp16 C[N*N];

void smol_matmul() {
  rand_array(A,N*N);
  rand_array(B,N*N);

  print_mat(A,N,N);
  printf("\n");
  print_mat(B,N,N);
  printf("\n");

  AMX_SET();

  for (uint64_t i = 0; i < N; i++) {
    AMX_LDX(PMASK & (uint64_t)(B + N*i) | (i << 56));
    AMX_LDY(PMASK & (uint64_t)(A + N*i) | (i << 56));
    AMX_FMA16((i*64) | ((i*64) << 10));
  }

  for (uint64_t i = 0; i < N; i++)
    AMX_STZ((PMASK & (uint64_t)(C+N*i)) | (2*i << 56));

  AMX_CLR();

  print_mat(C,N,N);
}

#define ITERATIONS 100000000

void accumulators() {
  uint64_t start, end;
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(&timebase_info);

  AMX_SET();

  start = mach_absolute_time(); 
  for (uint64_t i = 0; i < ITERATIONS; i++) {
    AMX_FMA16(0);
    AMX_FMA16(1 << 20); // about same throughput if we remove this line due to independence of z-registers
  }
  end = mach_absolute_time(); 

  AMX_CLR();

  uint64_t ns = (end-start)*timebase_info.numer/timebase_info.denom;
  double s = ns*1e-9;

  printf("%f\n", s);
}

int main() {
  smol_matmul();
  accumulators();
  return 0;
}
