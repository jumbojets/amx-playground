#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "amx.h"
#include "util.h"

// mini matrix multiplication
#define N 8

_Float16 A[N*N]; // assume that it is transposed
_Float16 B[N*N];
_Float16 C[N*N];

void smol_matmul() {
  rand_array(A,N*N);
  rand_array(B,N*N);

  print_mat(A,N,N);
  printf("\n");
  print_mat(B,N,N);
  printf("\n");

  AMX_SET();

  for (uint64_t i = 0; i < N; i++) {
    AMX_LDX(B+N*i, i, 0);
    AMX_LDY(A+N*i, i, 0);
    AMX_FMA16(i*64, i*64, 0, 0);
  }

  for (uint64_t i = 0; i < N; i++)
    AMX_STZ(C+N*i, 2*i, 0);

  AMX_CLR();

  print_mat(C,N,N);
}

#define ITERATIONS 100000000

void accumulators() {
  uint64_t start, end;

  AMX_SET();

  start = clock_gettime_nsec_np(CLOCK_REALTIME);
  for (uint64_t i = 0; i < ITERATIONS; i++) {
    AMX_FMA16(0, 0, 0, 0);
    AMX_FMA16(0, 0, 1, 0); // same throughput if we remove this line due to independence of z-registers
  }
  end = clock_gettime_nsec_np(CLOCK_REALTIME);

  AMX_CLR();

  double s = (end-start)*1e-9;

  printf("%f\n", s);
}

int main() {
  smol_matmul();
  accumulators();
  return 0;
}
