#include <stdint.h>
#include <stdlib.h>
#include "amx.h"
#include "util.h"

// small enough layer is chosen to keep weights in register without fuss and tiling
// the idea expressed by this code generalizes
#define N 8

#define RELU(x) (((x) > 0) ? (x) : 0)

int16_t L1Weights[N*N];
int16_t L1Bias[N];
int16_t L2Weights[N*N];
int16_t L2Bias[N];
int16_t X[N];

void layer_inplace(int16_t weights[N*N], int16_t bias[N*N], int16_t x[N]) {
  // must take a trip through memory to access RELU
  AMX_SET();

  AMX_LDY(PMASK & (uint64_t)x);
  AMX_LDZ(PMASK & (uint64_t)bias);
  for (uint64_t i = 0; i < N; i++) {
    AMX_LDX((PMASK & (uint64_t)weights) | (i << 3));
    AMX_MAC16((i*64) << 10);
  }
  AMX_STZ(PMASK & (uint64_t)x);

  AMX_CLR();

  for (int i = 0; i < N; i++)
    x[i] = RELU(x[i]);
}

int main() {
  rand_array(L1Weights,N*N);
  rand_array(L1Bias,N);
  rand_array(L2Weights,N*N);
  rand_array(L2Bias,N);
  rand_array(X,N);

  layer_inplace(L1Weights,L1Bias,X);
  layer_inplace(L2Weights,L2Bias,X);

  print_mat(X,1,N);
}
