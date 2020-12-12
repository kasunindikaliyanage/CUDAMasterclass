#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"

__global__ void atomics(int *shared_var, int iters)
{
	for (int i = 0; i < iters; i++)
	{
		atomicAdd(shared_var, 1);
	}
}

__global__ void unsafe(int *shared_var, int iters)
{
	for (int i = 0; i < iters; i++)
	{
		int old = *shared_var;
		*shared_var = old + 1;
	}
}

//int main(int argc, char **argv)
//{
//	int N = 64;
//	int block = 32;
//	int runs = 30;
//	int iters = 100000;
//	int r;
//	int *d_shared_var;
//	int h_shared_var_atomic, h_shared_var_unsafe;
//	int *h_values_read;
//
//	gpuErrchk(cudaMalloc((void **)&d_shared_var, sizeof(int)));
//
//	double atomic_mean_time = 0;
//	double unsafe_mean_time = 0;
//	clock_t ops_start, ops_end;
//
//	for (r = 0; r < runs; r++)
//	{
//		gpuErrchk(cudaMemset(d_shared_var, 0x00, sizeof(int)));
//	
//		ops_start = clock();
//		atomics <<< N / block, block >>>(d_shared_var,iters);
//		gpuErrchk(cudaDeviceSynchronize());
//		ops_end = clock();
//		atomic_mean_time += ops_end - ops_start;
//
//		gpuErrchk(cudaMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int),
//			cudaMemcpyDeviceToHost));
//		gpuErrchk(cudaMemset(d_shared_var, 0x00, sizeof(int)));
//		
//		ops_start = clock();
//		unsafe <<< N / block, block >>>(d_shared_var,iters);
//		gpuErrchk(cudaDeviceSynchronize());
//		ops_end = clock();
//		unsafe_mean_time += ops_end - ops_start;
//		
//		gpuErrchk(cudaMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int),
//			cudaMemcpyDeviceToHost));
//	}
//
//	atomic_mean_time = atomic_mean_time / CLOCKS_PER_SEC;
//	unsafe_mean_time = unsafe_mean_time / CLOCKS_PER_SEC;
//
//	printf("In total, %d runs using atomic operations took %f s\n",
//		runs, atomic_mean_time);
//	printf("  Using atomic operations also produced an output of %d\n",
//		h_shared_var_atomic);
//	printf("In total, %d runs using unsafe operations took %f s\n",
//		runs, unsafe_mean_time);
//	printf("  Using unsafe operations also produced an output of %d\n",
//		h_shared_var_unsafe);
//
//	return 0;
//}
