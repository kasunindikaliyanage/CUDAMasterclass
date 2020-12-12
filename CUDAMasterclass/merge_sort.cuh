#ifndef MERGE_SORT_CUH
#define MERGE_SORT_CUH

#include "common.h"
#include "cuda_common.cuh"

void mergesort_cpu(int *,int, int);
void mergesort_gpu(int *, int, int);

#endif