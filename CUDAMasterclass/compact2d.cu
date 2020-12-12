#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "common.h"
#include "cuda_common.cuh"

#define BLOCK_SIZE 64

__global__ void scan_for_compact2d(int * input, int * output_index_array, int* auxiliry_array, int input_size)
{
	int idx = threadIdx.x;
	int gid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ int local_input[BLOCK_SIZE];

	if (input[gid] >0)
	{
		local_input[idx] = 1;
	}
	else
	{
		local_input[idx] = 0;
	}

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
		local_input[idx] = 0;
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
		//printf("%d \n", auxiliry_array[blockIdx.x]);
	}
	output_index_array[gid] = local_input[idx];
}

__global__ void scan_summation_for_compact2d(int * output_index_array, int * auxiliry_array, int input_size)
{
	int idx = threadIdx.x;
	int gid = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ int local_input[BLOCK_SIZE];

	local_input[idx] = output_index_array[gid];
	__syncthreads();

	for (int i = 0; i < blockIdx.x; i++)
	{
		local_input[idx] += auxiliry_array[i];
	}

	output_index_array[gid] = local_input[idx];
	//printf("gid : %d, value : %d \n", gid, output_index_array[gid]);
}

//CSR compact of 2D matrix
__global__ void compact2d_1D_array(int * input, int * output, int * output_column_index_array,	
	int * output_row_index_array, int * prev_output_index_array,int* auxiliry_array, int array_size)
{
	int gid = blockDim.x*blockIdx.x + threadIdx.x;

	//TO DO handle when gid ==0
	//this is very unefficient in memory management
	if (gid > 0 && gid < array_size)
	{
		printf("gid : %d , index :%d , value : %d, prev_value : %d \n",gid, prev_output_index_array[gid], input[gid], input[gid-1]);
		if (prev_output_index_array[gid] != prev_output_index_array[gid - 1])
		{
			//printf("gid : %d , index :%d , value : %d, prev_value : %d \n",gid, output_index_array[gid], input[gid], input[gid-1]);
			output[prev_output_index_array[gid]] = input[gid - 1];
			output_column_index_array[prev_output_index_array[gid]] = (gid - 1)% blockDim.x;
		}

		int colum_index = gid / (blockDim.x  - 1);
		int condition = gid % (blockDim.x - 1);

		if (condition == 0)
		{
			printf("column index : %d --- row length : %d \n", condition, prev_output_index_array[gid]);
			if (gid == 0)
			{
				output_row_index_array[0] = 0;
			}
			else
			{
				output_row_index_array[colum_index] = prev_output_index_array[gid];
				//output_row_index_array[colum_index] = auxiliry_array[colum_index];
			}
		}
	}
}

void run_compact2d_using_1D()
{
	int columns = 64;
	int rows = 3;

	int input_size = rows*columns;
	int input_byte_size = input_size * sizeof(int);
	int * h_input = get_matrix(rows, columns);

	int row_bytes = sizeof(int) * (rows+1);

	dim3 block(BLOCK_SIZE);
	dim3 grid(input_size / block.x);
	int aux_byte_size = sizeof(int)*grid.x;

	int *h_ref, *h_aux_ref, *h_output, *h_column_index, *h_row_index;
	h_ref = (int*)malloc(input_byte_size);
	h_aux_ref = (int*)malloc(aux_byte_size);
	h_row_index = (int*)malloc(row_bytes);;
	
	int * d_input, *d_output_index_array,
		*d_aux, *d_sum_input, *d_sum_aux, *d_output,*d_column_index_array, *d_row_index_array;

	gpuErrchk(cudaMalloc((int**)&d_input, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_output_index_array, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_column_index_array, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_row_index_array, row_bytes));
	gpuErrchk(cudaMalloc((int**)&d_aux, aux_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_sum_input, input_byte_size));
	gpuErrchk(cudaMalloc((int**)&d_sum_aux, aux_byte_size));

	gpuErrchk(cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice));
	scan_for_compact2d << <grid, block >> > (d_input,d_output_index_array, d_aux, input_size);
	gpuErrchk(cudaDeviceSynchronize());

	//gpuErrchk(cudaMemcpy(d_sum_input, d_output_index_array, input_byte_size, cudaMemcpyDeviceToDevice));
	gpuErrchk(cudaMemcpy(d_sum_aux, d_aux, aux_byte_size, cudaMemcpyDeviceToDevice));
	scan_summation_for_compact2d << <grid, block >> > (d_output_index_array, d_sum_aux, input_size);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_ref, d_output_index_array, input_byte_size, cudaMemcpyDeviceToHost));

	int compact_output_size = h_ref[input_size - 1]+1;
	int compact_output_byte_size = sizeof(float)*compact_output_size;

	h_output = (int*)malloc(compact_output_byte_size);
	gpuErrchk(cudaMalloc((int**)&d_output, compact_output_byte_size));

	h_column_index = (int*)malloc(compact_output_byte_size);

	compact2d_1D_array << <grid, block >> > (d_input, d_output, d_column_index_array, d_row_index_array
		,d_output_index_array, d_sum_aux,input_size);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_output, d_output, compact_output_byte_size, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_column_index, d_column_index_array, compact_output_byte_size, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_row_index, d_row_index_array, row_bytes, cudaMemcpyDeviceToHost));
	
	for (int i = 0; i<compact_output_size; i++)
	{
		printf("value : %d - col index : %d \n", h_output[i], h_column_index[i]);
	}

	for (int i = 0; i < rows; i++)
	{
		printf("%d,",h_row_index[i]);
	}

	printf("\n");

	cudaFree(d_sum_input);
	cudaFree(d_sum_aux);
	cudaFree(d_input);
	cudaFree(d_aux);

	free(h_input);
	free(h_aux_ref);
	free(h_ref);
}

//int main()
//{
//	
//	run_compact2d_using_1D();
//	system("pause");
//	return 0;
//}