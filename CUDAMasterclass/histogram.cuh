#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include "common.h"
#include "cuda_common.cuh"

void histogram_cpu(int *, int*, int);
void histogram_gpu(int *, int*, int);

#endif
