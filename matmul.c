// Benchmarked on M1 Max 
// L1(performance core): 192+128KB per core
// L1(efficiency core):  128+64KB per core
// L2(performance):      24MB
// L2(efficiency):       4MB
// L3:                   48MB

// TODO:
// * more cache awareness
// * load 128 bytes into consecutive AMX registers (must be 128 byte aligned)
// * multithreaded

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "amx.h"
#include "util.h"

#define N 1024

// assume A is transposed to access columns as rows
__fp16 At[N*N] __attribute__ ((aligned (64)));
__fp16  B[N*N] __attribute__ ((aligned (64)));
__fp16  C[N*N] __attribute__ ((aligned (64)));

void matmul() {
  for (int i = 0; i < N; i+=32) {     // C tile row
    for (int j = 0; j < N; j+=64) {   // C tile col. process two at once on seperate z-accumulators
      AMX_SET(); // z-register reset to zero

      for (int k = 0; k < N; k+=32) { // down cols of At and B

        // the X,Y registers can only hold 8 partial rows of 32 int16s
        for (int rb = 0; rb < 32; rb+=8) {
          #pragma clang loop unroll(full)
          for (uint64_t r = 0; r < 8; r++) {
            AMX_LDY((PMASK & (uint64_t)(At + (k+rb+r)*N + i)) | (r << 56));
            uint64_t xr1 = (2*r)%8; // TODO: should be cleaner way of doing this
            uint64_t xr2 = (2*r+1)%8;
            AMX_LDX((PMASK & (uint64_t)(B  + (k+rb+r)*N + j)) | (xr1 << 56));
            AMX_LDX((PMASK & (uint64_t)(B  + (k+rb+r)*N + j+32)) | (xr2 << 56));
            AMX_FMA16((r*64) | (xr1*64 << 10));
            AMX_FMA16((r*64) | (xr2*64 << 10) | (1 << 20));
          }
        }
      }

      #pragma clang loop unroll_count(8)
      for (uint64_t r = 0; r < 32; r++) {
        AMX_STZ((PMASK & (uint64_t)(C + (i+r)*N + j)) | (2*r << 56));
        AMX_STZ((PMASK & (uint64_t)(C + (i+r)*N + j+32)) | ((2*r+1) << 56));
      }

      AMX_CLR();
    }
  }
}

#define ITERATIONS 10
#define CHECK_EQUIV 0
#define EPSILON 1

int main() {
  srand(time(NULL));

  uint64_t start, end;
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(&timebase_info);

  for (int i = 0; i < ITERATIONS; i++) {
    rand_array(At,N*N);
    rand_array(B,N*N);
    memset(C,0,N*N*sizeof(int16_t));

    start = mach_absolute_time();
    matmul();
    end = mach_absolute_time();

    uint64_t ns = (end-start)*timebase_info.numer/timebase_info.denom;
    double gflop = (2.0*N*N*N)*1e-9;
    double s = ns*1e-9;
    printf("%f GFLOP/s -- %.2f ms\n", gflop/s, s*1e3);

#if CHECK_EQUIV
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        __fp16 real = 0;
        for (int k = 0; k < N; k++)
          real += At[k*N+i] * B[k*N+j];
        if ((C[i*N+j] - real) > EPSILON) {
          printf("not equivalent at (%d, %d): %f != %f\n", i, j, C[i*N+j], real);
          return 1;
        }
      }
    }
    printf("equivalence test passed\n");
#endif

  }

  return 0;
}
