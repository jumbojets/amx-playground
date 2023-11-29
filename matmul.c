// Benchmarked on M1 Max 
// L1(performance core): 192+128KB per core
// L1(efficiency core):  128+64KB per core
// L2(performance):      24MB
// L2(efficiency):       4MB
// L3:                   48MB

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "amx.h"

#define PMASK 0xffffffffffffff
#define N 64

int16_t At[N*N]; // Assume A is transposed so can access columns as rows lol
int16_t B[N*N];
int16_t C[N*N];

void rand_array(int16_t arr[N*N]) {
  for (int i = 0; i < N*N; i++)
    arr[i] = rand() % 10;
}

void print_mat(int16_t mat[N*N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      printf("%d, ", mat[i*N+j]);
    printf("\n");
  }
}

int main() {
  rand_array(At);
  rand_array(B);

  for (int zi = 0; zi < N/32; zi++) {     // zreg tile row. register tile holds 2x matrices of 32x32 2-byte elements. only use 1x for now
    for (int zj = 0; zj < N/32; zj++) {   // zreg tile col

      // accumulate z here. start at zero
      AMX_SET();

      for (int zk = 0; zk < N/32; zk++) { // go down At and B
        // the X,Y registers can only hold 8 partial rows of 32 int16s
        // so its furthur divided into 16 (4x4) tiles of 8x8

        for (int xyi = 0; xyi < 32/8; xyi++) {
          for (int xyj = 0; xyj < 4; xyj++) {
            for (int xyk = 0; xyk < 4; xyk++) {

              // load the 8 partial rows into both x and y
              #pragma clang loop unroll(full)
              for (uint64_t pr = 0; pr < 8; pr++) {
                AMX_LDX((PMASK & (uint64_t)(At + (zi*32 + xyi*8 + pr)*N) + 2*8*(zj*32 + xyj*8 + pr)) | (pr << 56));
                AMX_LDX((PMASK & (uint64_t)(B  + (zi*32 + xyi*8 + pr)*N) + 2*8*(zj*32 + xyj*8 + pr)) | (pr << 56));
              }

              // can put this in above for loop, but im not using x,y registers well...
              #pragma clang loop unroll(full)
              for (uint64_t pr = 0; pr < 8; pr++) 
                AMX_MAC16(((pr*64) << 10 | (pr*64)));

            }
          }
        }
      }

      #pragma clang loop unroll_count(8)
      for (uint64_t r = 0; r < 32; r++)
        AMX_STZ((PMASK & ((uint64_t)C + (zi*32 + r)*N + 2*32*(zj*32 + r))) | ((2*r) << 56));
        
      AMX_CLR();
    }
  }

  print_mat(C);

  // check if equivalent naive matrix multiplication implementation
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
