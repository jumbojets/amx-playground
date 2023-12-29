// Benchmarked on M1 Max 
// L1(performance core): 192+128KB per core
// L1(efficiency core):  128+64KB per core
// L2(performance):      24MB
// L2(efficiency):       4MB
// L3:                   48MB

// TODO:
// * use other half of z-register 
// * more cache awareness
// * load 128 bytes into consecutive AMX registers (must be 128 byte aligned)
// * multithreaded

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "amx.h"
#include "util.h"

#define N 1024

// assume A is transposed to access columns as rows
int16_t At[N*N] __attribute__ ((aligned (64)));
int16_t  B[N*N] __attribute__ ((aligned (64)));
int16_t  C[N*N] __attribute__ ((aligned (64)));

void matmul() {
  for (int i = 0; i < N/32; i++) {     // C tile row
    for (int j = 0; j < N/32; j++) {   // C tile col

      AMX_SET(); // z-register reset to zero

      for (int k = 0; k < N/32; k++) { // down cols of At and B

        // the X,Y registers can only hold 8 partial rows of 32 int16s
        for (int rb = 0; rb < 32/8; rb++) {
#pragma clang loop unroll(full)
          for (uint64_t r = 0; r < 8; r++) {
            AMX_LDY((PMASK & (uint64_t)(At + (k*32 + rb*8 + r)*N + i*32)) | (r << 56));
            AMX_LDX((PMASK & (uint64_t)(B  + (k*32 + rb*8 + r)*N + j*32)) | (r << 56));
            AMX_MAC16((r*64) | (r*64 << 10));
          }
        }
      }

#pragma clang loop unroll_count(8)
      for (uint64_t r = 0; r < 32; r++)
        AMX_STZ((PMASK & (uint64_t)(C + (i*32 + r)*N + j*32)) | (r*2 << 56));

      AMX_CLR();
    }
  }
}

#define ITERATIONS 5
#define CHECK_EQUIV 0

int main() {
  srand(time(NULL));

  uint64_t start, end;
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(&timebase_info);
  
  for (int i = 0; i < ITERATIONS; i++) {
    rand_array(At,N*N);
    rand_array(B,N*N);

    start = mach_absolute_time();
    matmul();
    end = mach_absolute_time();

    uint64_t ns = (end-start)*timebase_info.numer/timebase_info.denom;
    double gop = (2.0*N*N*N)*1e-9;
    double s = ns*1e-9;
    printf("%f GOP/s -- %.2f ms\n", gop/s, s*1e3);

#if CHECK_EQUIV
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
#endif

  }
  return 0;
}
