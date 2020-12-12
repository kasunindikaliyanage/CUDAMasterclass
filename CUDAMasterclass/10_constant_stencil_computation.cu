#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define c0 1
#define c1 2
#define c2 3
#define c3 4
#define c4 5

#define RADIUS 4

#define BDIM 128

//constant memory declaration
__constant__ int coef[9];

//stencil calculation in host side
void host_const_calculation(int * in, int * out, int size)
{
	for (int i = 0; i < size; i++)
	{

		if (i < RADIUS)
		{
			out[i] = in[i + 4] * c0
				+ in[i + 3] * c1
				+ in[i + 2] * c2
				+ in[i + 1] * c3
				+ in[i] * c4;

			if (i == 3)
			{
				out[i] += in[2] * c3;
				out[i] += in[1] * c2;
				out[i] += in[0] * c1;
			}
			else if (i == 2)
			{
				out[i] += in[1] * c3;
				out[i] += in[0] * c2;
			}
			else if (i == 1)
			{
				out[i] += in[0] * c3;
			}
		}
		else if ((i + RADIUS) >= size)
		{
			out[i] = in[i - 4] * c0
				+ in[i - 3] * c1
				+ in[i - 2] * c2
				+ in[i - 1] * c3
				+ in[i] * c4;
			
			if (i == size - 4)
			{
				out[i] += in[size - 3] * c3;
				out[i] += in[size - 2] * c2;
				out[i] += in[size - 1] * c1;
			}
			else if (i == size -3)
			{
				out[i] += in[size - 2] * c3;
				out[i] += in[size - 1] * c2;
			}
			else if (i == size - 2)
			{
				out[i] += in[size - 1] * c3;
			}
		}
		else
		{
			out[i] = (in[i - 4] + in[i + 4])*c0
				+ (in[i - 3] + in[i + 3])*c1
				+ (in[i - 2] + in[i + 2])*c2
				+ (in[i - 1] + in[i + 1])*c3
				+ in[i] * c4;
		}
	}
}

//setting up constant memory from host
void setup_coef_1()
{
	const int h_coef[] = { c0,c1,c2,c3,c4,c3,c2,c1,c0 };
	cudaMemcpyToSymbol(coef, h_coef, (9) * sizeof(float));
}

__global__ void constant_stencil_smem_test(int * in, int * out, int size)
{
	//shared mem declaration
	__shared__ int smem[BDIM + 2 * RADIUS];

	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	int bid = blockIdx.x;
	int num_of_blocks = gridDim.x;

	int value = 0;

	if (gid < size)
	{
		//index with offset
		int sidx = threadIdx.x + RADIUS;

		//load data to shared mem
		smem[sidx] = in[gid];

		if (bid != 0 && bid != (num_of_blocks - 1))
		{
			if (threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = in[gid - RADIUS];
				smem[sidx + BDIM] = in[gid + BDIM];
			}
		}
		else if (bid == 0)
		{
			if (threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = 0;
				smem[sidx + BDIM] = in[gid + BDIM];
			}
		}
		else
		{
			if (threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = in[gid - RADIUS];
				smem[sidx + BDIM] = 0;
			}
		}

		// wait untill all the threads in block finish storing smem
		__syncthreads();

		value += smem[sidx - 4] * coef[0];
		value += smem[sidx - 3] * coef[1];
		value += smem[sidx - 2] * coef[2];
		value += smem[sidx - 1] * coef[3];
		value += smem[sidx - 0] * coef[4];
		value += smem[sidx + 1] * coef[5];
		value += smem[sidx + 2] * coef[6];
		value += smem[sidx + 3] * coef[7];
		value += smem[sidx + 4] * coef[8];

		out[gid] = value;
	}
}

//int main(int argc, char ** argv)
//{
//	int size = 1 << 22;
//	int byte_size = sizeof(int) * size;
//	int block_size = BDIM;
//
//	int * h_in, *h_out, *h_ref;
//
//	h_in = (int*)malloc(byte_size);
//	h_out = (int*)malloc(byte_size);
//	h_ref = (int*)malloc(byte_size);
//
//	initialize(h_in, size, INIT_ONE);
//
//	int * d_in, *d_out;
//	cudaMalloc((void**)&d_in, byte_size);
//	cudaMalloc((void**)&d_out, byte_size);
//
//	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
//	cudaMemset(d_out, 0, byte_size);
//
//	setup_coef_1();
//
//	dim3 blocks(block_size);
//	dim3 grid(size / blocks.x);
//
//	constant_stencil_smem_test << < grid, blocks >> > (d_in, d_out, size);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	host_const_calculation(h_in, h_out, size);
//
//	compare_arrays(h_ref, h_out, size);
//
//	cudaFree(d_out);
//	cudaFree(d_in);
//	free(h_ref);
//	free(h_out);
//	free(h_in);
//
//	return 0;
//}