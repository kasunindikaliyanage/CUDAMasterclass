#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 128
#define FULL_MASK 0xffffffff

template<unsigned int iblock_size>
__global__ void reduction_smem_benchmark(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;

	smem[tid] = i_data[tid];

	__syncthreads();

	// in-place reduction in shared memory   
	if (blockDim.x >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();

	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = smem[0];
	}
}

template<unsigned int iblock_size>
__global__ void reduction_smem_warp_shfl(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;

	smem[tid] = i_data[tid];

	__syncthreads();

	// in-place reduction in shared memory   
	if (blockDim.x >= 1024 && tid < 512)
		smem[tid] += smem[tid + 512];
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
		smem[tid] += smem[tid + 256];
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
		smem[tid] += smem[tid + 128];
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
		smem[tid] += smem[tid + 64];
	__syncthreads();

	if (blockDim.x >= 64 && tid < 32)
		smem[tid] += smem[tid + 32];
	__syncthreads();

	int local_sum = smem[tid];

	//unrolling warp
	if (tid < 32)
	{
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 16);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 8);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 4);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 2);
		local_sum += __shfl_down_sync(FULL_MASK, local_sum, 1);
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = local_sum;
	}
}


//int main(int argc, char ** argv)
//{
//	printf("Running parallel reduction with complete unrolling kernel \n");
//
//	int kernel_index = 0;
//
//	if (argc > 1)
//	{
//		kernel_index = 1;
//	}
//
//	int size = 1 << 25;
//	int byte_size = size * sizeof(int);
//	int block_size = BLOCK_SIZE;
//
//	int * h_input, *h_ref;
//	h_input = (int*)malloc(byte_size);
//
//	initialize(h_input, size);
//
//	int cpu_result = reduction_cpu(h_input, size);
//
//	dim3 block(block_size);
//	dim3 grid((size / block_size));
//
//	printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
//
//	int temp_array_byte_size = sizeof(int)* grid.x;
//
//	h_ref = (int*)malloc(temp_array_byte_size);
//
//	int * d_input, *d_temp;
//
//	printf(" \nreduction with shared memory\n ");
//	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
//
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,cudaMemcpyHostToDevice));
//	
//	reduction_smem_benchmark <1024> << < grid, block >> > (d_input, d_temp, size);
//
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	int gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	compare_results(gpu_result, cpu_result);
//
//	//warp shuffle implementation
//	printf(" \nreduction with warp shuffle instructions \n ");
//	
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));
//
//	reduction_smem_warp_shfl <1024> << < grid, block >> > (d_input, d_temp, size);
//
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	compare_results(gpu_result, cpu_result);
//
//	gpuErrchk(cudaFree(d_input));
//	gpuErrchk(cudaFree(d_temp));
//	free(h_input);
//	free(h_ref);
//
//	gpuErrchk(cudaDeviceReset());
//	return 0;
//}