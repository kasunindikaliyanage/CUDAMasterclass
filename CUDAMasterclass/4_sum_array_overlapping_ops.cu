#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

void sumArraysOnHostx(int *A, int *B, int *C, const int N)
{
	for (int idx = 0; idx < N; idx++)
		C[idx] = A[idx] + B[idx];
}

__global__ void sum_array_overlap(int * a, int * b, int * c, int N)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < N)
	{
		c[gid] = a[gid] + b[gid];
	}
}

//int main()
//{
//	int size = 1 << 25;
//	int block_size = 128;
//
//	//number of bytes needed to hold element count
//	size_t NO_BYTES = size * sizeof(int);
//
//	int const NUM_STREAMS = 8;
//	int ELEMENTS_PER_STREAM = size / NUM_STREAMS;
//	int BYTES_PER_STREAM = NO_BYTES / NUM_STREAMS;
//
//	// host pointers
//	int *h_a, *h_b, *gpu_result, *cpu_result;
//
//	//allocate memory for host size pointers
//	cudaMallocHost((void**)&h_a,NO_BYTES);
//	cudaMallocHost((void**)&h_b, NO_BYTES);
//	cudaMallocHost((void**)&gpu_result, NO_BYTES);
//
//	cpu_result = (int *)malloc(NO_BYTES);
//
//	//initialize h_a and h_b arrays randomly
//	initialize(h_a, INIT_ONE_TO_TEN);
//	initialize(h_b, INIT_ONE_TO_TEN);
//
//	//summation in CPU
//	sumArraysOnHostx(h_a, h_b, cpu_result, size);
//
//	int *d_a, *d_b, *d_c;
//	cudaMalloc((int **)&d_a, NO_BYTES);
//	cudaMalloc((int **)&d_b, NO_BYTES);
//	cudaMalloc((int **)&d_c, NO_BYTES);
//
//	cudaStream_t streams[NUM_STREAMS];
//
//	for (int i = 0; i < NUM_STREAMS; i++)
//	{
//		cudaStreamCreate(&streams[i]);
//	}
//
//	//kernel launch parameters
//	dim3 block(block_size);
//	dim3 grid(ELEMENTS_PER_STREAM/block.x + 1);
//
//	int offset = 0;
//
//	for (int  i = 0; i < NUM_STREAMS; i++)
//	{
//		offset = i * ELEMENTS_PER_STREAM;
//		cudaMemcpyAsync(&d_a[offset], &h_a[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice,streams[i]);
//		cudaMemcpyAsync(&d_b[offset], &h_b[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice,streams[i]);
//
//		sum_array_overlap << <grid, block, 0, streams[i] >> > (&d_a[offset], &d_b[offset], &d_c[offset], size);
//
//		cudaMemcpyAsync(&gpu_result[offset], &d_c[offset], BYTES_PER_STREAM, cudaMemcpyDeviceToHost,streams[i]);
//	}
//	
//	for (int i = 0; i < NUM_STREAMS; i++)
//	{
//		cudaStreamDestroy(streams[i]);
//	}
//
//	cudaDeviceSynchronize();
//
//	//validity check
//	compare_arrays(cpu_result, gpu_result, size);
//
//	cudaFree(d_a);
//	cudaFree(d_b);
//	cudaFree(d_c);
//
//	cudaFreeHost(h_a);
//	cudaFreeHost(h_b);
//	cudaFreeHost(gpu_result);
//	free(cpu_result);
//
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//}