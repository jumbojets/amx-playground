// Benchmarked on M1 Max 
// L1(performance core): 192+128KB per core
// L1(efficiency core):  128+64KB per core
// L2(performance):      24MB
// L2(efficiency):       4MB
// L3:                   48MB

// TODO:
// * use other half of z-register 
// * more cache awareness
// * multithreaded

#include <mach/mach_time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "amx.h"
#include "util.h"

#define N 1024

// assume A is transposed to access columns as rows
int16_t At[N*N] __attribute__ ((aligned (64)));
int16_t  B[N*N] __attribute__ ((aligned (64)));
int16_t  C[N*N] __attribute__ ((aligned (64)));

void matmul() {
  for (int zi = 0; zi < N/32; zi++) {     // z-reg tile row. register tile holds 2x matrices of 32x32 2-byte elements. only use 1x for now
    for (int zj = 0; zj < N/32; zj++) {   // z-reg tile col

      AMX_SET(); // z-register reset to zero

      for (int zk = 0; zk < N/32; zk++) { // go down At and B

        // the X,Y registers can only hold 8 partial rows of 32 int16s
        for (int xy = 0; xy < 32/8; xy++) {
          // load the 8 partial rows into both x and y and execute
#pragma clang loop unroll(full)
          for (uint64_t pr = 0; pr < 8; pr++) {
            AMX_LDY((PMASK & (uint64_t)(At + (zk*32 + pr + 8*xy)*N + (zi*32))) | (pr << 56));
            AMX_LDX((PMASK & (uint64_t)(B  + (zk*32 + pr + 8*xy)*N + (zj*32))) | (pr << 56));
            AMX_MAC16(((pr*64) << 10 | (pr*64)));
          }
        }
      }

#pragma clang loop unroll_count(8)
      for (uint64_t r = 0; r < 32; r++)
        AMX_STZ((PMASK & (uint64_t)(C + (zi*32 + r)*N + (zj*32))) | (2*r << 56));

      AMX_CLR();
    }
  }
}

int main() {
  uint64_t start, end;
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(&timebase_info);

  rand_array(At,N*N);
  rand_array(B,N*N);

  start = mach_absolute_time();
  matmul();
  end = mach_absolute_time();

  uint64_t ns = (end-start)*timebase_info.numer/timebase_info.denom;
  double gflop = (2.0*N*N*N)*1e-9;
  double s = ns*1e-9;
  printf("%f GFLOP/s -- %.2f ms\n", gflop/s, s*1e3);

  // check if equivalent to naive matrix multiplication implementation
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int16_t real = 0;
      for (int k = 0; k < N; k++)
        real += At[k*N+i] * B[k*N+j];
      if (C[i*N+j] != real) {
        printf("not equivalent at (%d, %d): %hd != %hd\n", i, j, C[i*N+j], real);
        return 1;
      }
    }
  }

  return 0;
}
