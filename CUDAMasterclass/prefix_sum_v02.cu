#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "cuda_common.cuh"


#define BLOCK_SIZE  128

//this implementation works only for 1D grid with 1D blocks
//auxliry array should be size of number of blocks
__global__ void scan_efficient_1G(int * input, int* auxiliry_array ,int input_size)
{
	//check creation of "idx variable is good or bad". if we create idx this way it should be on registers
	//where threadIdx variable lies. threadIdx is it in global memory or registers
	int idx = threadIdx.x;
	int gid = blockDim.x*blockIdx.x + threadIdx.x;

	extern __shared__ int local_input[];

	local_input[idx] = input[gid];
	__syncthreads();

	// reduction phase
	// this can be optimized check wether global memory access for "d" or calculation here is better
	int d = ceilf(log2f(BLOCK_SIZE));
	int denominator = 0;
	int offset = 0;

	//reduction should happen per block
	for (int i = 1; i <= d; i++)
	{
		denominator = 1 << i;
		offset = 1 << (i - 1);
		if (((idx + 1) % denominator) == 0)
		{
			local_input[idx] += local_input[idx - offset];
		}
		__syncthreads();
	}

	////end of reduction phase

	//// start of  down-sweep phase
	if (idx == (BLOCK_SIZE - 1))
	{
		local_input[idx] = local_input[0];
	}

	int temp = 0;
	int sawp_aux = 0;

	for (int i = d; i > 0; i--)
	{
		temp = 1 << i;
		if ((idx != 0) && (idx + 1) % temp == 0)
		{
			sawp_aux = local_input[idx];
			local_input[idx] += local_input[idx - (temp / 2)];
			local_input[idx - (temp / 2)] = sawp_aux;
		}
		__syncthreads();
	}
	
	//can this be add to if condition at the begining of the down sweep phase 
	if (idx == (BLOCK_SIZE - 1))
	{
		auxiliry_array[blockIdx.x] = local_input[idx];
		//printf("aux value for - blockIdx =%d, %d \n", blockIdx.x ,auxiliry_array[blockIdx.x]);
	}
	input[gid] = local_input[idx];
}

__global__ void scan_summation(int * input, int * auxiliry_array, int input_size)
{
	int idx = threadIdx.x;
	int gid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ int local_input[BLOCK_SIZE];

	local_input[idx] = input[gid];
	__syncthreads();

	for (int i = 0; i < blockIdx.x; i++)
	{
		local_input[idx] += auxiliry_array[i];
	}

	input[gid] = local_input[idx];
}

//this implementation works only for 1D grid with 1 block
__global__ void scan_efficient_1G1B(int * input, int input_size)
{
	int idx = threadIdx.x;
	int gid = blockDim.x*blockIdx.x + threadIdx.x;

	// reduction phase
	int d = log2f(input_size);
	for (int i = 1; i <= d  ; i++)
	{
		int denominator = 1<<i;
		int offset = 1<<(i-1);
		if (((gid + 1) % denominator) == 0)
		{
			input[gid] += input[gid - offset];
		}
		__syncthreads();
	}

	////end of reduction phase
	printf("After reduction  GID : %d - %d \n", gid, input[gid]);

	//// start of  down-sweep phase
	if (gid==input_size-1)
	{
		input[gid] = input[0];
	}

	int temp =0;
	int sawp_aux = 0;

	for (int i = d; i > 0; i--)
	{
		temp = 1<<i;
		if ((gid != 0 ) && (gid + 1) % temp ==0)
		{
			sawp_aux = input[gid];
			input[gid] += input[gid - (temp / 2)];
			input[gid - (temp / 2)] = sawp_aux;
		}
	}
	//printf("final  GID : %d - %d \n",gid ,input[gid]);
}

void run_scan_efficient_1G(int argc, char **argv)
{
	int input_size = 1 << 23;

	if (argc > 1)
		input_size = 1 << atoi(argv[1]);


	int input_byte_size = input_size * sizeof(int);
	dim3 block(BLOCK_SIZE);
	dim3 grid(input_size / block.x);
	int aux_byte_size = sizeof(int)*grid.x;

	int* h_input, *h_ref, *h_aux_ref;
	h_input = (int*)malloc(input_byte_size);
	h_ref = (int*)malloc(input_byte_size);
	h_aux_ref = (int*)malloc(aux_byte_size);

	for (int i = 0; i < input_size; i++)
	{
		h_input[i] = 1;
	}

	printf("lauching kernel with grid.x = %d, block.x = %d, memory = %d MB \n", grid.x,block.x, (input_byte_size/(1024*1024)));

	int * d_input, *d_aux, *d_sum_input, *d_sum_aux;
	gpuErrchk(cudaMalloc((int**)&d_input, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_aux, aux_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_sum_input, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_sum_aux, aux_byte_size));

	gpuErrchk(cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice));
	scan_efficient_1G << <grid, block, sizeof(int)*BLOCK_SIZE >> > (d_input, d_aux, input_size);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(d_sum_input, d_input, input_byte_size, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(d_sum_aux, d_aux, aux_byte_size, cudaMemcpyDeviceToDevice));
	scan_summation << <grid, block >> > (d_sum_input, d_sum_aux, input_size);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_ref, d_sum_input, input_byte_size, cudaMemcpyDeviceToHost));

	/*for (int i = 0; i < input_size; i++)
	{
		printf("%d,", h_ref[i]);
	}
*/
	printf("\n");
	printf("final element value - %d \n", h_ref[input_size - 1]);

	cudaFree(d_sum_input);
	cudaFree(d_sum_aux);
	cudaFree(d_input);
	cudaFree(d_aux);

	free(h_input);
	free(h_aux_ref);
	free(h_ref);

	cudaDeviceReset();
}

//int main(int argc, char **argv)
//{
//	//query_device();
//	//run_scan_efficient_1G(argc, argv);
//
//
//
//	system("pause");
//	return 0;
//}