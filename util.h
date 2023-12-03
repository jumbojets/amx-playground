#pragma once
#include <stdint.h>
#include <stdio.h>

#define PMASK 0xffffffffffffff

void rand_array(int16_t *arr, int size) {
  for (int i = 0; i < size; i++)
    arr[i] = rand() % 10;
}

void print_mat(int16_t *arr, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++)
      printf("%d, ", arr[i*rows+j]);
    printf("\n");
  }
}
