#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "amx.h"

#define PMASK 0xffffffffffffff
#define N 32

int16_t X[N];
int16_t Y[N];
int16_t Z[N*N];

void rand_array(int16_t arr[N]) {
  for (int i = 0; i < N; i++)
    arr[i] = rand() % 10;
}

void print_mat(int16_t *arr, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++)
      printf("%d, ", arr[i*rows+j]);
    printf("\n");
  }
}

int main() {
  srand(time(NULL));

  rand_array(X);
  rand_array(Y);

  print_mat(X,1,N);
  print_mat(Y,1,N);

  AMX_SET();

  AMX_LDX(PMASK & (uint64_t)X);
  AMX_LDY(PMASK & (uint64_t)Y);
  
  // AMX_MAC16(1ULL << 63); // vector op
  // AMX_STZ(PMASK & (uint64_t)Z);

  AMX_MAC16(0LL); // matrix op (outer product)

  for (uint64_t i = 0; i < N; i++)
    AMX_STZ((PMASK & ((uint64_t)Z+2*N*i)) | (2*i << 56));

  AMX_CLR();

  print_mat(Z,N,N);
}
