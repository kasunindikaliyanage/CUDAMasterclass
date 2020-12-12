#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void incr(int *ptr)
{
	/*int temp = *ptr;
	temp = temp + 1;
	*ptr = temp;*/
	atomicAdd(ptr,1);
}

//int main()
//{
//	int value = 0;	
//	int SIZE = sizeof(int);
//	int ref = -1;
//
//	int *d_val;
//	cudaMalloc((void**)&d_val, SIZE);
//	cudaMemcpy(d_val, &value, SIZE, cudaMemcpyHostToDevice);
//	incr << <1, 32 >> > (d_val);
//	cudaDeviceSynchronize();
//	cudaMemcpy(&ref,d_val,SIZE, cudaMemcpyDeviceToHost);
//
//	printf("Updated value : %d \n",ref);
//
//	cudaDeviceReset();
//	return 0;
//}