#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void test_sum_array_for_memory(float* a, float* b, float *c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (gid < size)
		c[gid] = a[gid] + b[gid];
}

//int main(int argc, char** argv)
//{
//	printf("Runing 1D grid \n");
//	int size = 1 << 22;
//	int block_size = 128;
//
//	if (argc > 1)
//		block_size = 1 << atoi(argv[1]);
//
//	printf("Entered block size : %d \n", block_size);
//
//	unsigned int byte_size = size * sizeof(float);
//
//	printf("Input size : %d \n", size);
//
//	float * h_a, *h_b, *h_ref;
//	h_a = (float*)malloc(byte_size);
//	h_b = (float*)malloc(byte_size);
//	h_ref = (float*)malloc(byte_size);
//
//
//	if (!h_a)
//		printf("host memory allocation error \n");
//
//	for (size_t i = 0; i < size; i++)
//	{
//		h_a[i] = i % 10;
//		h_b[i] = i % 7;
//	}
//
//	dim3 block(block_size);
//	dim3 grid((size + block.x - 1) / block.x);
//
//	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
//		grid.x, grid.y, grid.z, block.x, block.y, block.z);
//
//	float *d_a, *d_b, *d_c;
//
//	gpuErrchk(cudaMalloc((void**)&d_a, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_b, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_c, byte_size));
//	gpuErrchk(cudaMemset(d_c, 0, byte_size));
//
//	gpuErrchk(cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice));
//
//	test_sum_array_for_memory << <grid, block >> > (d_a, d_b, d_c, size);
//
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_c, byte_size, cudaMemcpyDeviceToHost));
//
//	cudaFree(d_c);
//	cudaFree(d_b);
//	cudaFree(d_a);
//
//	free(h_ref);
//	free(h_b);
//	free(h_a);
//}
