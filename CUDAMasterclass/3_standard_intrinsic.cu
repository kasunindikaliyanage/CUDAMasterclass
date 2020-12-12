#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void standard(float *ptr)
{
	*ptr = powf(*ptr, 2.0f);
}

__global__ void intrinsic(float *ptr)
{ 
	*ptr = __powf(*ptr, 2.0f);
}

//int main()
//{
//	float value = 23;	
//	int SIZE = sizeof(float);
//
//	float *d_val;
//	cudaMalloc((void**)&d_val, SIZE);
//	cudaMemcpy(d_val, &value, SIZE, cudaMemcpyHostToDevice);
//	standard << <1, 1 >> > (d_val);
//	intrinsic << <1, 1 >> > (d_val);
//	cudaDeviceSynchronize();
//
//	cudaDeviceReset();
//	return 0;
//}