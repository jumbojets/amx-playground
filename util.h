#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "amx.h"

#define PMASK 0xffffffffffffff

void rand_array(__fp16 *arr, int size) {
  for (int i = 0; i < size; i++)
    arr[i] = (__fp16)((float)rand() / RAND_MAX - 0.5);
}

void print_mat(__fp16 *arr, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++)
      printf("%f, ", arr[i*cols+j]);
    printf("\n");
  }
}

void read_x(__fp16 ret[256]) {
  for (uint64_t i = 0; i < 8; i++)
    AMX_STX(PMASK & (uint64_t)(ret+i*32) | (i << 56));
}

void read_y(__fp16 ret[256]) {
  for (uint64_t i = 0; i < 8; i++)
    AMX_STY(PMASK & (uint64_t)(ret+i*32) | (i << 56));
}

void read_z(__fp16 ret[2048]) {
  for (uint64_t i = 0; i < 64; i++)
    AMX_STZ(PMASK & (uint64_t)(ret+i*32) | (i << 56));
}
