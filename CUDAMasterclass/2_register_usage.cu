#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void  register_usage_test(int * results, int size)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	int x1 = 3465;
	int x2 = 1768;
	int x3 = 453;
	int x7 = 3465;
	int x5 = 1768;
	int x6 = 453;
	int x4 = x1 + x2 + x3 + x7 + x5 + x6;

	if (gid < size)
	{
		results[gid] =  x4;
	}
}

//int main()
//{
//	int size = 1 << 22;
//	int byte_size = sizeof(int)*size;
//
//	int * h_ref = (int*) malloc(byte_size);
//	int * d_results;
//	cudaMalloc((void**)&d_results, byte_size);
//	cudaMemset(d_results, 0, byte_size);
//
//	dim3 blocks(128);
//	dim3 grid((size+blocks.x-1)/blocks.x);
//
//	printf("launching the kernel \n");
//	register_usage_test << <grid,blocks >> > (d_results, size);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_results, byte_size, cudaMemcpyDeviceToHost);
//	printf("Results have arrived \n");
//
//	int sum = 0;
//
//	for (int i = 0; i < size; i++)
//	{
//		sum += h_ref[i];
//	}
//
//	printf("final sum : %d \n",sum);
//
//	return 0;
//}



