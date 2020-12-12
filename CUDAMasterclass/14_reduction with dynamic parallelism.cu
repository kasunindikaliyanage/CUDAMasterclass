#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"
#include "common.h"

//__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata,
//	unsigned int isize)
//{
//	int tid = threadIdx.x;
//
//	int *idata = g_idata + blockIdx.x*blockDim.x;
//	int *odata = &g_odata[blockIdx.x];
//
//	// stop condition   
//	if (isize == 2 && tid == 0)
//	{
//		g_odata[blockIdx.x] = idata[0] + idata[1];
//		return;
//	}
//
//	// nested invocation   
//	int istride = isize >> 1;
//	if (istride > 1 && tid < istride)
//	{
//		// in place reduction    
//		idata[tid] += idata[tid + istride];
//	}
//
//	// sync at block level   
//	__syncthreads();
//
//	// nested invocation to generate child grids 
//	if (tid == 0)
//	{
//		gpuRecursiveReduce << <1, istride >> > (idata, odata, istride);
//		cudaDeviceSynchronize();
//	}
//
//	// sync at block level again 
//	__syncthreads();
//}


//int main(int argc, char ** argv)
//{
//	printf("Running parallel reduction with interleaved pairs kernel \n");
//
//	int size = 1 << 22;
//	int byte_size = size * sizeof(int);
//	int block_size = 512;
//	clock_t gpu_start, gpu_end,cpu_start, cpu_end;
//
//	int * h_input, *h_ref;
//	h_input = (int*)malloc(byte_size);
//	initialize(h_input, size, INIT_RANDOM);
//
//	cpu_start = clock();
//	int cpu_result = reduction_cpu(h_input, size);
//	cpu_end = clock();
//
//	dim3 block(block_size);
//	dim3 grid(size / block.x);
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
//	gpu_start = clock();
//
//	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
//	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size,
//		cudaMemcpyHostToDevice));
//
//	gpuRecursiveReduce <<< grid, block >>> (d_input, d_temp,block_size);
//
//	gpuErrchk(cudaDeviceSynchronize());
//	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));
//
//	int gpu_result = 0;
//	for (int i = 0; i < grid.x; i++)
//	{
//		gpu_result += h_ref[i];
//	}
//	gpu_end = clock();
//	print_time_using_host_clock(gpu_start, gpu_end);
//
//	printf("CPU kernel execution time : %4.6f \n",
//		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
//
//	compare_results(gpu_result, cpu_result);
//
//	gpuErrchk(cudaFree(d_input));
//	gpuErrchk(cudaFree(d_temp));
//	free(h_input);
//	free(h_ref);
//
//	cudaDeviceReset();
//	return 0;
//}