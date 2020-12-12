#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#define ARRAY_SIZE 32

__global__ void test_shfl_broadcast_32(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_sync(0xffffffff, x, 3, 32);
	out[threadIdx.x] = y;
}

__global__ void test_shfl_broadcast_16(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_sync(0xffffffff, x, 3, 16);
	out[threadIdx.x] = y;
}

__global__ void test_shfl_up(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_up_sync(0xffffffff, x, 3);
	out[threadIdx.x] = y;
}

__global__ void test_shfl_down(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_down_sync(0xffffffff, x, 3);
	out[threadIdx.x] = y;
}

//__global__ void test_shfl_shift_around(int * in, int *out, int offset)
//{
//	int x = in[threadIdx.x];
//	int y = __shfl_sync(0xffffffff, x, threadIdx.x + offset);
//	out[threadIdx.x] = y;
//}

__global__ void test_shfl_xor_butterfly(int * in, int *out)
{
	int x = in[threadIdx.x];
	int y = __shfl_xor_sync(0xffffffff, x, 1, 32);
	out[threadIdx.x] = y;
}


//int main(int argc, char ** argv)
//{
//	int size = ARRAY_SIZE;
//	int byte_size = size * sizeof(int);
//
//	int * h_in = (int*)malloc(byte_size);
//	int * h_ref = (int*)malloc(byte_size);
//
//	for (int i = 0; i < size; i++)
//	{
//		h_in[i] = i;
//	}
//
//	int * d_in, *d_out;
//
//	cudaMalloc((int **)&d_in, byte_size);
//	cudaMalloc((int **)&d_out, byte_size);
//
//	dim3 block(size);
//	dim3 grid(1);
//
//	//broadcast 32
//	printf("shuffle broadcast 32 \n");
//	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
//	test_shfl_broadcast_32 << < grid, block >> > (d_in, d_out);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	print_array(h_in, size);
//	print_array(h_ref, size);
//
//	//broadcast 16
//	printf("shuffle broadcast 16 \n");
//	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
//	test_shfl_broadcast_16 << < grid, block >> > (d_in, d_out);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	print_array(h_in, size);
//	print_array(h_ref, size);
//	printf("\n");
//
//	//up
//	printf("shuffle up \n");
//	cudaMemset(d_out, 0, byte_size);
//	test_shfl_up << < grid, block >> > (d_in, d_out);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	print_array(h_in, size);
//	print_array(h_ref, size);
//	printf("\n");
//
//	//down
//	printf("shuffle down \n");
//	cudaMemset(d_out, 0, byte_size);
//	test_shfl_down << < grid, block >> > (d_in, d_out);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	print_array(h_in, size);
//	print_array(h_ref, size);
//	printf("\n");
//
//	//shift around
//	printf("shift around \n");
//	cudaMemset(d_out, 0, byte_size);
//	test_shfl_shift_around << < grid, block >> > (d_in, d_out, 2);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	print_array(h_in, size);
//	print_array(h_ref, size);
//	printf("\n");
//
//	//shuffle xor butterfly
//	printf("shuffle xor butterfly \n");
//	cudaMemset(d_out, 0, byte_size);
//	test_shfl_xor_butterfly << < grid, block >> > (d_in, d_out);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	print_array(h_in, size);
//	print_array(h_ref, size);
//	printf("\n");
//
//	cudaFree(d_out);
//	cudaFree(d_in);
//	free(h_ref);
//	free(h_in);
//
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//}