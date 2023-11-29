// Benchmarked on M1 max
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
#define N 4

int16_t X[N*N];
int16_t Y[N*N];
int16_t Z[N*N];
int16_t Zreal[N*N];

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
  rand_array(X);
  rand_array(Y);
  
  // cache tile (fit in L1)
  // 2B*
  // amx register z tile (64x64 bytes)

  AMX_SET();

  for (int rti = 0; rti < N/32; rti++) {   // reg tile i
    for (int rtj = 0; rtj < N/32; rtj++) { // reg tile j
      // (i,j)th (32x32) tile of Z

      // iterate over A and B

    }
  }

  AMX_CLR();

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      Zreal[i*N+j] = 0;
      for (int k = 0; k < N; k++)
        Zreal[i*N+j] += X[i*N+k] * Y[k*N+j];
      if (Z[i*N+j] != Zreal[i*N+j]) {
        printf("not equivalent\n");
        return 1;
      }
    }
  }

  return 0;
}
