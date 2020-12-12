#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#define SHARED_ARRAY_SIZE 128

__global__ void smem_static_test(int * in, int * out, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int smem[SHARED_ARRAY_SIZE];

	if (gid < size)
	{
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}

__global__ void smem_dynamic_test(int * in, int * out, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int smem[];

	if (gid < size)
	{
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}

//int main(int argc, char ** argv)
//{
//	int size = 1 << 22;
//	int block_size = SHARED_ARRAY_SIZE;
//	bool dynamic = false;
//
//	if (argc > 1)
//	{
//		dynamic = atoi(argv[1]);
//	}
//
//	//number of bytes needed to hold element count
//	size_t NO_BYTES = size * sizeof(int);
//
//	// host pointers
//	int *h_in, *h_ref, *d_in, *d_out;
//
//	//allocate memory for host size pointers
//	h_in = (int *)malloc(NO_BYTES);
//	h_ref = (int *)malloc(NO_BYTES);
//
//	initialize(h_in, size, INIT_ONE_TO_TEN);
//
//	cudaMalloc((int **)&d_in, NO_BYTES);
//	cudaMalloc((int **)&d_out, NO_BYTES);
//
//	//kernel launch parameters
//	dim3 block(block_size);
//	dim3 grid((size / block.x) + 1);
//
//	cudaMemcpy(d_in, h_in, NO_BYTES, cudaMemcpyHostToDevice);
//
//	if (!dynamic)
//	{
//		printf("Static smem kernel \n");
//		smem_static_test << <grid, block >> > (d_in, d_out, size);
//	}
//	else
//	{
//		printf("Dynamic smem kernel \n");
//		smem_dynamic_test << <grid, block, sizeof(int)*  SHARED_ARRAY_SIZE >> > (d_in, d_out, size);
//	}
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, NO_BYTES, cudaMemcpyDeviceToHost);
//
//	cudaFree(d_in);
//	cudaFree(d_out);
//
//	free(h_in);
//	free(h_ref);
//
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//}