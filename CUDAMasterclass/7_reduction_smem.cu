#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 1024

template<unsigned int iblock_size>
__global__ void reduction_gmem_benchmark(int * input,int * temp, int size)
{
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;

	//manual unrolling depending on block size
	if (iblock_size >= 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];

	__syncthreads();

	if (iblock_size >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];

	__syncthreads();

	if (iblock_size >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];

	__syncthreads();

	if (iblock_size >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];

	__syncthreads();

	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = i_data;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}

template<unsigned int iblock_size>
__global__ void reduction_smem(int * input, int * temp, int size)
{
	__shared__ int smem[BLOCK_SIZE];
	int tid = threadIdx.x;
	int * i_data = input + blockDim.x * blockIdx.x;

	smem[tid] = i_data[tid];

	__syncthreads();

	//manual unrolling depending on block size
	if (iblock_size >= 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];

	__syncthreads();

	if (iblock_size >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];

	__syncthreads();

	if (iblock_size >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];

	__syncthreads();

	if (iblock_size >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];

	__syncthreads();

	//unrolling warp
	if (tid < 32)
	{
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}


//int main(int argc, char ** argv)
//{
//    printf("Running parallel reduction with complete unrolling kernel \n");
//
//	int kernel_index = 0;
//
//	if (argc >1)
//	{
//		kernel_index = 1;
//	}
//
//	int size = 1 << 22;
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
//	int temp_array_byte_size = sizeof(int)* grid.x;
//
//	h_ref = (int*)malloc(temp_array_byte_size);
//
//	int * d_input, *d_temp;
//	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
//
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//	if (kernel_index == 0)
//	{
//		printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
//
//		switch (block_size)
//		{
//		case 1024:
//			reduction_gmem_benchmark <1024> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 512:
//			reduction_gmem_benchmark <512> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 256:
//			reduction_gmem_benchmark <256> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 128:
//			reduction_gmem_benchmark <128> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 64:
//			reduction_gmem_benchmark <64> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		}
//	}
//	else
//	{
//		printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
//
//		switch (block_size)
//		{
//		case 1024:
//			reduction_smem <1024> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 512:
//			reduction_smem <512> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 256:
//			reduction_smem <256> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 128:
//			reduction_smem <128> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		case 64:
//			reduction_smem <64> << < grid, block >> > (d_input, d_temp, size);
//			break;
//		}
//	}
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
//	gpuErrchk(cudaFree(d_input));
//	gpuErrchk(cudaFree(d_temp));
//	free(h_input);
//	free(h_ref);
//
//	gpuErrchk(cudaDeviceReset());
//	return 0;
//}