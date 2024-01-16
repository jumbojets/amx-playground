// gcc -O3 -o accelerate accelerate.c -I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers -framework Accelerate && ./accelerate
#include <cblas.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4096
#define ITERATIONS 10

void rand_array(float *arr, int size) {
  for (int i = 0; i < size; i++)
    arr[i] = (float)rand() / RAND_MAX - 0.5;
}

float A[N*N] __attribute__ ((aligned (64)));
float B[N*N] __attribute__ ((aligned (64)));
float C[N*N] __attribute__ ((aligned (64)));

int main() {
  uint64_t start, end;

  rand_array(A, N*N);
  rand_array(B, N*N);
  rand_array(C, N*N);

  start = clock_gettime_nsec_np(CLOCK_REALTIME);

  for (uint64_t i = 0; i < ITERATIONS; i++)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 1.0f, C, N);

  end = clock_gettime_nsec_np(CLOCK_REALTIME);

  double gflop = (2.0*N*N*N + N*N)*1e-9*ITERATIONS;
  double s = (end-start)*1e-9;
  printf("%f GFLOP/s -- %.2f ms\n", gflop/s, s*1e3);

  return 0;
}
