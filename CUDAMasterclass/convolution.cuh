#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "cuda_common.cuh"
//#include "cuda.h"

// CPU convolution algorithm for 1D input array
void convolution_cpu_1d(float*, float*, float*, int, int);

//naive gpu implementation of 1D convolution algorithm
void convolution_gpu_1d_launch(float*, float*, float*, int, int,int);