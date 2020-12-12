#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "common.h"

#define SECTION_SIZE 64

void prefix_sum_cpu(int * input, int * output ,int input_size)
{
	output[0] = input[0];
	for (int i = 1; i < input_size; i++)
	{
		output[i] = input[i] + output[i - 1];
	}
}

//imagine 1D grid with 1D blocks with shared memory
__global__ void prefix_sum_gpu_shared_mem(int* input, int * intermideate_results,int input_size)
{
	int gid = blockDim.x*blockIdx.x + threadIdx.x;
	int condition_value = blockDim.x / 2;

	__shared__ int block_inputs[SECTION_SIZE];
	block_inputs[threadIdx.x] = input[gid];
	__syncthreads();

	int temp = 0;
	if (gid < input_size)
	{
		for (int stride = 1; stride <= condition_value; stride = stride << 1)
		{
			temp = threadIdx.x - stride;
			if ((temp) >= 0)
			{
				block_inputs[threadIdx.x] += block_inputs[temp];
			}
			__syncthreads();
		}

		input[gid] = block_inputs[threadIdx.x];
		__syncthreads();

		if (threadIdx.x == (blockDim.x - 1))
		{
			intermideate_results[blockIdx.x] = input[gid];
			//printf("%d - %d \n", blockIdx.x,intermideate_results[blockIdx.x]);
			//printf("gid - %d, block_id -%d, sum - %d \n", gid, blockIdx.x,  block_inputs[threadIdx.x]);
		}

		//int block_indicator = gid / blockDim.x;
		if (blockIdx.x > 0)
		{
			for (int i = 0; i < blockIdx.x; i++)
			{
				input[gid] += intermideate_results[i];
			}
		}
	}
}

//works of only 1D grid with 1D blocks
//look at the gird and block configuration before run this
__global__ void prefix_sum_gpu(int*  input, int input_size)
{
	int gid = blockDim.x*blockIdx.x + threadIdx.x;
	int condition_value = blockDim.x / 2;
	for (int stride = 1; stride <= condition_value; stride = stride << 1)
	{
		if ((threadIdx.x - stride) >= 0)
		{
			input[gid] = input[gid] + input[gid - stride];
		}
		__syncthreads();
	}
	//printf("%d,", input[gid]);
}

void run_prefix_sum_on_cpu(int input_shift)
{
	int * input, *output;
	int input_size = 1 << input_shift;
	int input_bytes_size = sizeof(int)*input_size;

	input = (int*)malloc(input_bytes_size);
	output = (int*)malloc(input_bytes_size);

	initialize(input,input_size, INIT_ONE);
	prefix_sum_cpu(input, output,input_size);
	printf("Prifix array in CPU \n");
	printf("Final value : %d ", output[input_size - 1]);

	free(output);
	free(input);
}

void run_prefix_sun_on_gpu(int input_shift)
{
	int grid_size;
	int * h_input, *h_output;
	int input_size = 1 << input_shift;
	int input_bytes_size = sizeof(int)*input_size;

	h_input = (int*)malloc(input_bytes_size);
	initialize(h_input,input_size,INIT_ONE);
	h_output = (int*)malloc(input_bytes_size);

	//we need to handle scenarios where input_size / SECTION_SIZE <0
	if (input_size / SECTION_SIZE <= 1)
	{
		grid_size = 1;
	}
	else
	{
		grid_size = input_size / SECTION_SIZE;
	}

	dim3 grid(grid_size);
	dim3 block(input_size/grid.x);

	printf("Number of blocks : %d \n", grid.x);
	printf("Number of threads in a block : %d \n", block.x);

	int* d_input, *d_intermideate_array;
	cudaMalloc((int**)&d_input, input_bytes_size);
	cudaMalloc((int**)&d_intermideate_array, sizeof(int)*grid_size);

	cudaMemcpy(d_input, h_input, input_bytes_size, cudaMemcpyHostToDevice);
	prefix_sum_gpu_shared_mem <<< grid,block >>> (d_input, d_intermideate_array,input_size);
	//prefix_sum_gpu << < grid, block >> > (d_input, input_size);
	cudaDeviceSynchronize();

	cudaError error = cudaMemcpy(h_output, d_input, input_bytes_size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("%s,%s", cudaGetErrorString(error),cudaGetErrorName(error));
	}

	printf("Prifix array in GPU \n");
	//my_print_array(h_input,input_size);
	//my_print_array(h_output, input_size);
	printf("Final value : %d ", h_output[input_size - 1]);

	cudaFree(d_input);
	free(h_output);
	free(h_input);
}

//int main()
//{
//	int input_shift = 14;
//	run_prefix_sum_on_cpu(input_shift);
//	run_prefix_sun_on_gpu(input_shift);
//	system("pause");
//	return 0;
//}