#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//warm up kernel
__global__ void warmup_kernel(int * input,int * temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		if (tid % (2 * offset) == 0)
		{
			input[gid] += input[gid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = input[gid];
	}
}

//reduction neighbored pairs kernel
__global__ void redunction_neighbored_pairs_1(int * input,
	int * temp, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		if (tid % (2 * offset) == 0)
		{
			input[gid] += input[gid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = input[gid];
	}
}

//reduction neighbored pairs imporved kernel
__global__ void reduction_neighbored_pairs_improved_1(
	int * int_array, int * temp_array, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	//local data block pointer
	int * i_data = int_array + blockDim.x * blockIdx.x;

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		int index = 2 * offset * tid;

		if (index < blockDim.x)
		{
			i_data[index] += i_data[index + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		temp_array[blockIdx.x] = int_array[gid];
	}
}

//interleaved pairs kernel
__global__ void reduction_interleaved_pairs_1(int * int_array,
	int * temp_array, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid > size)
		return;

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
	{
		if (tid < offset)
		{
			int_array[gid] += int_array[gid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		temp_array[blockIdx.x] = int_array[gid];
	}
}

// reduction with unrolling 2 blocks
__global__ void reduction_interleaved_unrolling_blocks2_1(int * input,
	int * temp, int size)
{
	int tid = threadIdx.x;

	//start index for this thread
	int index = blockDim.x * blockIdx.x * 2 + threadIdx.x;

	//local index for this block
	int * i_data = input + blockDim.x * blockIdx.x * 2;

	//unrolling two blocks
	if ((index + blockDim.x)< size)
	{
		input[index] += input[index + blockDim.x];
	}

	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0;
		offset = offset / 2)
	{
		if (tid < offset)
		{
			i_data[tid] += i_data[tid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}

// reduction with unrolling 8 blocks
__global__ void reduction_interleaved_unrolling_blocks8_1(int * input,
	int * temp, int size)
{
	int tid = threadIdx.x;

	//element index for this thread
	int index = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	//local data pointer
	int * i_data = input + blockDim.x * blockIdx.x * 8;

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

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
	{
		if (tid < offset)
		{
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		temp[blockIdx.x] = i_data[0];
	}
}

// reduction warp unrolling with 8 blocks unrolling
__global__ void reduction_kernel_interleaved_warp_unrolling8_1(int * input,
	int * temp_array, int size)
{
	int tid = threadIdx.x;

	//element index for this thread
	int index = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	//local data pointer
	int * i_data = input + blockDim.x * blockIdx.x * 8;

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

	for (int offset = blockDim.x / 2; offset >= 64;
		offset = offset / 2)
	{
		if (tid < offset)
		{
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}

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
		temp_array[blockIdx.x] = i_data[0];
	}
}

// reduction complete unrolling
__global__ void reduction_kernel_complete_unrolling8_1(int * input,
	int * temp, int size)
{
	int tid = threadIdx.x;
	int index = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	int * i_data = input + blockDim.x * blockIdx.x * 8;

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

	//complete unrolling manually

	//if the block dim == 1024 
	if (blockDim.x == 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];
	__syncthreads();


	// warp unrolling
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

// reduction with complete unrolling and template functions
template<unsigned int iblock_size>
__global__ void reduction_kernel_complete_template8_1(int * input,
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
//	printf("Performance comparison of reduction for 128Mb data aray \n");
//
//	int size = 1 << 25;
//	int byte_size = size * sizeof(int);
//	int block_size = 1024;
//
//	clock_t cpu_start, cpu_end, gpu_start, gpu_end;
//
//	int * h_input, *h_ref;
//	h_input = (int*)malloc(byte_size);
//
//	initialize(h_input, size);
//
//	cpu_start = clock();
//	int cpu_result = reduction_cpu(h_input, size);
//	cpu_end = clock();
//
//	dim3 block(block_size);
//	dim3 grid((size / block_size));
//
//	//printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);
//	printf("CPU execution time : %4.6f \n",
//		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
//
//	int temp_array_byte_size = sizeof(int)* grid.x;
//
//	h_ref = (int*)malloc(temp_array_byte_size);
//
//	int * d_input, *d_temp;
//	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
//	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));
//
//	int gpu_result = 0;
//	dim3 new_grid2(grid.x / 2);
//	dim3 new_grid8(grid.x / 8);
//
//	// 0 warm up kernel
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	warmup_kernel << <grid, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//
//
//	// 1 naive reduction implementation : neighbored pairs approach
//	printf("1. Naive neighbored pairs approach \n");
//	gpu_start = clock();
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	redunction_neighbored_pairs_1 << <grid, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	gpu_end = clock();
//	//compare_results(gpu_result, cpu_result);
//	print_time_using_host_clock(gpu_start, gpu_end);
//	printf("\n");
//
//	// 2. improved neighbored pairs approach
//	printf("2. Improved neighbored pairs approach \n");
//	gpu_start = clock();
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	reduction_neighbored_pairs_improved_1 << <grid, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	gpu_end = clock();
//	//compare_results(gpu_result, cpu_result);
//	print_time_using_host_clock(gpu_start, gpu_end);
//	printf("\n");
//
//	// 3. interleaved pairs approach
//	printf("3. Interleaved pairs approach \n");
//
//	gpu_start = clock();
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	reduction_interleaved_pairs_1 << <grid, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	gpu_end = clock();
//	//compare_results(gpu_result, cpu_result);
//	print_time_using_host_clock(gpu_start, gpu_end);
//	printf("\n");
//
//	// 4. 2 data blocks unrolled to one
//	printf("4. Data blocks unrolled 2 \n");
//
//	gpu_start = clock();
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	reduction_interleaved_unrolling_blocks2_1 << <new_grid2, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	gpu_end = clock();
//	//compare_results(gpu_result, cpu_result);
//	print_time_using_host_clock(gpu_start, gpu_end);
//	printf("\n");
//	
//	// 5. 8 data blocks unrolled to one
//	printf("5. Data blocks unrolled 8 \n");
//
//	gpu_start = clock();
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	reduction_interleaved_unrolling_blocks8_1 << <new_grid8, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	gpu_end = clock();
//	//compare_results(gpu_result, cpu_result);
//	print_time_using_host_clock(gpu_start, gpu_end);
//	printf("\n");
//
//	// 6. warp unrolling for 8 blocks unrolloed 
//	printf("6. warp unrolling and 8 data block unrolling \n");
//
//	gpu_start = clock();
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	reduction_kernel_interleaved_warp_unrolling8_1<< <new_grid8, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//
//	gpu_end = clock();
//	//compare_results(gpu_result, cpu_result);
//	print_time_using_host_clock(gpu_start, gpu_end);
//	printf("\n");
//
//	// 7. complete unrolling
//	printf("7. complete unrolling \n");
//	gpu_start = clock();
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	reduction_kernel_complete_unrolling8_1 << <new_grid8, block >> > (d_input, d_temp, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//	gpu_end = clock();
//
//	//compare_results(gpu_result, cpu_result);
//	print_time_using_host_clock(gpu_start, gpu_end);
//	printf("\n");
//
//	// 8. complete unrolling with templated functions 
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	switch (block_size)
//	{
//	case 1024:
//		reduction_kernel_complete_template8_1 <1024> << < new_grid8, block >> > (d_input, d_temp, size);
//		break;
//	case 512:
//		reduction_kernel_complete_template8_1 <512> << < new_grid8, block >> > (d_input, d_temp, size);
//		break;
//	case 256:
//		reduction_kernel_complete_template8_1 <256> << < new_grid8, block >> > (d_input, d_temp, size);
//		break;
//	case 128:
//		reduction_kernel_complete_template8_1 <128> << < new_grid8, block >> > (d_input, d_temp, size);
//		break;
//	}
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//	//compare_results(gpu_result, cpu_result);
//
//	gpuErrchk(cudaFree(d_input));
//	gpuErrchk(cudaFree(d_temp));
//	free(h_input);
//	free(h_ref);
//
//	gpuErrchk(cudaDeviceReset());
//	return 0;
//} 