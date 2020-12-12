#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"
#include <stdio.h>

//for random intialize
#include <stdlib.h>
#include <time.h>

//for memset
#include <cstring>

#include "common.h"

__global__ void sum_arrays_gpu(int * a, int * b, int* c, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size)
		c[index] = a[index] + b[index];
}

void sum_arrays_cpu(int * a, int * b, int * c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

//int main()
//{
//	int size = 1 << 25;
//	int block_size = 128;
//	cudaError error;
//
//	//number of bytes needed to hold element count
//	size_t NO_BYTES = size * sizeof(int);
//
//	// host pointers
//	int *h_a, *h_b, *gpu_result, *cpu_result;
//
//	//allocate memory for host size pointers
//	h_a = (int *)malloc(NO_BYTES);
//	h_b = (int *)malloc(NO_BYTES);
//	gpu_result = (int *)malloc(NO_BYTES);
//	cpu_result = (int *)malloc(NO_BYTES);
//
//	//initialize h_a and h_b arrays randomly
//	time_t t;
//	srand((unsigned)time(&t));
//
//	for (size_t i = 0; i < size; i++)
//	{
//		h_a[i] = (int)(rand() & 0xFF);
//		//h_a[i] = 1;
//	}
//
//	for (size_t i = 0; i < size; i++)
//	{
//		h_b[i] = (int)(rand() & 0xFF);
//		//h_b[i] = 2;
//	}
//
//	memset(gpu_result, 0, NO_BYTES);
//	memset(cpu_result, 0, NO_BYTES);
//
//	//summation in CPU
//	clock_t cpu_start, cpu_end;
//	cpu_start = clock();
//	sum_arrays_cpu(h_a, h_b, cpu_result, size);
//	cpu_end = clock();
//
//	int *d_a, *d_b, *d_c;
//	gpuErrchk(cudaMalloc((int **)&d_a, NO_BYTES));
//	gpuErrchk(cudaMalloc((int **)&d_b, NO_BYTES));
//	gpuErrchk(cudaMalloc((int **)&d_c, NO_BYTES));
//
//	//kernel launch parameters
//	dim3 block(block_size);
//	dim3 grid((size / block.x) + 1);
//
//	clock_t mem_htod_start, mem_htod_end;
//	mem_htod_start = clock();
//	gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));
//	mem_htod_end = clock();
//
//	//execution time measuring in GPU
//	clock_t gpu_start, gpu_end;
//	gpu_start = clock();
//
//	sum_arrays_gpu << <grid, block >> > (d_a, d_b, d_c, size);
//	gpuErrchk(cudaDeviceSynchronize());
//	gpu_end = clock();
//
//	clock_t mem_dtoh_start, mem_dtoh_end;
//	mem_dtoh_start = clock();
//	gpuErrchk(cudaMemcpy(gpu_result, d_c, NO_BYTES, cudaMemcpyDeviceToHost));
//	mem_dtoh_end = clock();
//
//	compare_arrays(cpu_result, gpu_result, size);
//
//	printf("CPU sum time : %4.6f \n",
//		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
//
//	printf("GPU kernel execution time sum time : %4.6f \n",
//		(double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
//
//	printf("Mem transfer host to device : %4.6f \n",
//		(double)((double)(mem_htod_end - mem_htod_start) / CLOCKS_PER_SEC));
//
//	printf("Mem transfer device to host : %4.6f \n",
//		(double)((double)(mem_dtoh_end - mem_dtoh_start) / CLOCKS_PER_SEC));
//
//	printf("Total GPU time : %4.6f \n",
//		(double)((double)((mem_htod_end - mem_htod_start)
//			+ (gpu_end - gpu_start)
//			+ (mem_dtoh_end - mem_dtoh_start)) / CLOCKS_PER_SEC));
//
//	gpuErrchk(cudaFree(d_a));
//	gpuErrchk(cudaFree(d_b));
//	gpuErrchk(cudaFree(d_c));
//
//	free(h_a);
//	free(h_b);
//	free(gpu_result);
//}

