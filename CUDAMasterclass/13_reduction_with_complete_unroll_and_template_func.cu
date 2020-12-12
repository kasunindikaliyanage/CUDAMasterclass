#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<unsigned int iblock_size>
__global__ void reduction_kernel_complete_template(int * input, 
	int * temp, int size)
{
	int tid = threadIdx.x;
	int index = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	int * i_data = input + blockDim.x * blockIdx.x * 8;
	
	//unrolling blocks
	if ((index + 7 * blockDim.x) < size)
	{
		int a1 = input[index];
		int a2 = input[index + blockDim.x];
		int a3 = input[index + 2 * blockDim.x];
		int a4 = input[index + 3 * blockDim.x];
		int a5 = input[index + 4 * blockDim.x];
		int a6 = input[index + 5 * blockDim.x];
		int a7 = input[index + 6 * blockDim.x];
		int a8 = input[index + 7 * blockDim.x];

		input[index] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}

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

//int main(int argc, char ** argv)
//{
//    printf("Running parallel reduction with complete unrolling kernel \n");
//
//	int size = 1 << 25;
//	int byte_size = size * sizeof(int);
//	int block_size = 128;
//
//	int * h_input, *h_ref;
//	h_input = (int*)malloc(byte_size);
//
//	initialize(h_input, size);
//
//	int cpu_result = reduction_cpu(h_input, size);
//
//	dim3 block(block_size);
//	dim3 grid((size / block_size) / 8);
//
//	printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
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
//
//	switch (block_size)
//	{
//	case 1024 :
//		reduction_kernel_complete_template <1024> <<< grid, block >> > (d_input, d_temp, size);
//		break;
//	case 512:
//		reduction_kernel_complete_template <512> << < grid, block >> > (d_input, d_temp, size);
//		break;
//	case 256:
//		reduction_kernel_complete_template <256> << < grid, block >> > (d_input, d_temp, size);
//		break;
//	case 128:
//		reduction_kernel_complete_template <128> << < grid, block >> > (d_input, d_temp, size);
//		break;
//	case 64:
//		reduction_kernel_complete_template <64> << < grid, block >> > (d_input, d_temp, size);
//		break;
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