#include <stdint.h>

#include "amx.h"
#include "util.h"

// small enough layer is chosen to keep weights in register without fuss and tiling
// the idea expressed by this code generalizes
#define N 8

int16_t L1Weights[N*N];
int16_t L1Bias[N];
int16_t L2Weights[N*N];
int16_t L2Bias[N];
int16_t X[N];

int main() {
  // two layer perceptron with RELU activation function
  // must take a trip through memory to access RELU via SIMD

  rand_array(L1Weights,N*N);
  rand_array(L1Bias,N);
  rand_array(L2Weights,N*N);
  rand_array(L2Bias,N);
  rand_array(X,N);

  AMX_SET();

  for (uint64_t i = 0; i < 8) {
    AMX_LDX(PMASK & (uint64_t)L1Weights);
    // load bias into Z
  }

  AMX_CLR();
}
