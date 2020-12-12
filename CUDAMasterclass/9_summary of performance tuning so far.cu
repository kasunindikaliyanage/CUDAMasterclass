#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"


__global__ void misaligned_read_benchmark(int* a, int* b, int *c, int size, int offset)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int k = gid + offset;

	if (k < size)
		c[gid] = a[k] + b[k];
}

__global__ void misaligned_read_unrolled4(int* a, int* b, int *c, int size, int offset)
{
	int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int k = i + offset;

	if (k + 3 * blockDim.x < size)
	{
		c[i] = a[k] + b[k];
		c[i + blockDim.x] = a[k + blockDim.x] + b[k + blockDim.x];
		c[i + 2* blockDim.x] = a[k + 2 * blockDim.x] + b[k + 2 *blockDim.x];
		c[i + 3* blockDim.x] = a[k + 3* blockDim.x] + b[k + 3* blockDim.x];
	}
}


//int main(int argc, char** argv)
//{
//	printf("Runing 1D grid \n");
//	int size = 1 << 11;
//	int block_size = 128;
//	unsigned int byte_size = size * sizeof(float);
//	int offset = 0;
//
//	if (argc > 1)
//		offset = atoi(argv[1]);
//
//	printf("Input size : %d , offset : %d \n", size, offset);
//
//	int * h_a, *h_b, *h_ref1, *h_ref2;
//	h_a = (int*)malloc(byte_size);
//	h_b = (int*)malloc(byte_size);
//	h_ref1 = (int*)malloc(byte_size);
//	h_ref2 = (int*)malloc(byte_size);
//
//
//	if (!h_a)
//		printf("host memory allocation error \n");
//
//	for (size_t i = 0; i < size; i++)
//	{
//		h_a[i] = i % 10;
//		h_b[i] = i % 7;
//	}
//
//	dim3 block(block_size);
//	dim3 grid((size) / block.x);
//
//	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
//		grid.x, grid.y, grid.z, block.x, block.y, block.z);
//
//	int *d_a, *d_b, *d_c;
//
//	cudaMalloc((void**)&d_a, byte_size);
//	cudaMalloc((void**)&d_b, byte_size);
//	cudaMalloc((void**)&d_c, byte_size);
//	cudaMemset(d_c, 0, byte_size);
//
//	cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);
//
//	clock_t gpu_start, gpu_end;
//	gpu_start = clock();
//
//	misaligned_read_benchmark << <grid, block >> > (d_a, d_b, d_c, size, offset);
//
//	cudaDeviceSynchronize();
//
//	gpu_end = clock();
//	print_time_using_host_clock(gpu_start, gpu_end);
//	cudaMemcpy(h_ref1, d_c, byte_size, cudaMemcpyDeviceToHost);
//
//	cudaMemset(d_c, 0, byte_size);
//	grid.x = grid.x / 4;
//
//	printf("Kernel is lauch with grid(%d,%d,%d) and block(%d,%d,%d) \n",
//		grid.x, grid.y, grid.z, block.x, block.y, block.z);
//
//	gpu_start = clock();
//	
//	misaligned_read_unrolled4 << <grid, block >> > (d_a, d_b, d_c, size, offset);
//	cudaDeviceSynchronize();
//
//	gpu_end = clock();
//	print_time_using_host_clock(gpu_start, gpu_end);
//
//	cudaMemcpy(h_ref2, d_c, byte_size, cudaMemcpyDeviceToHost);
//
//	compare_arrays(h_ref1, h_ref2, size);
//	//print_array(h_ref1, size);
//	//print_array(h_ref2, size);
//
//	cudaFree(d_c);
//	cudaFree(d_b);
//	cudaFree(d_a);
//	free(h_ref1);
//	free(h_ref2);
//	free(h_b);
//	free(h_a);
//}