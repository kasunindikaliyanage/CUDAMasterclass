#include <stdio.h>
#include <stdlib.h>
#include "time.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"

__global__ void standard_kernel(float a, float *out, int iters)
{
	int i;
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (tid == 0)
	{
		float tmp;

		for (i = 0; i < iters; i++)
		{
			tmp = powf(a, 2.0f);
		}

		*out = tmp;
	}
}

__global__ void intrinsic_kernel(float a, float *out, int iters)
{
	int i;
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (tid == 0)
	{
		float tmp;

		for (i = 0; i < iters; i++)
		{
			tmp = __powf(a, 2.0f);
		}

		*out = tmp;
	}
}

//int main(int argc, char **argv)
//{
//	int i;
//	int runs = 30;
//	int iters = 1000;
//
//	float *d_standard_out, h_standard_out;
//	gpuErrchk(cudaMalloc((void **)&d_standard_out, sizeof(float)));
//
//	float *d_intrinsic_out, h_intrinsic_out;
//	gpuErrchk(cudaMalloc((void **)&d_intrinsic_out, sizeof(float)));
//
//	float input_value = 8181.25;
//
//	double mean_intrinsic_time = 0.0;
//	double mean_standard_time = 0.0;
//
//	clock_t ops_start, ops_end;
//
//	for (i = 0; i < runs; i++)
//	{
//		ops_start = clock();
//		standard_kernel << <1, 32 >> >(input_value, d_standard_out, iters);
//		gpuErrchk(cudaDeviceSynchronize());
//		ops_end = clock();
//		mean_standard_time += ops_end - ops_start;
//
//		ops_start = clock();
//		intrinsic_kernel << <1, 32 >> >(input_value, d_intrinsic_out, iters);
//		gpuErrchk(cudaDeviceSynchronize());
//		ops_end = clock();
//		mean_intrinsic_time += ops_end - ops_start;
//	}
//
//	mean_intrinsic_time = mean_intrinsic_time / CLOCKS_PER_SEC;
//	mean_standard_time = mean_standard_time / CLOCKS_PER_SEC;
//
//	gpuErrchk(cudaMemcpy(&h_standard_out, d_standard_out, sizeof(float),
//		cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float),
//		cudaMemcpyDeviceToHost));
//	float host_value = powf(input_value, 2.0f);
//
//	printf("Host calculated\t\t\t%f\n", host_value);
//	printf("Standard Device calculated\t%f\n", h_standard_out);
//	printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
//	printf("Host equals Standard?\t\t%s diff=%e\n",
//		host_value == h_standard_out ? "Yes" : "No",
//		fabs(host_value - h_standard_out));
//	printf("Host equals Intrinsic?\t\t%s diff=%e\n",
//		host_value == h_intrinsic_out ? "Yes" : "No",
//		fabs(host_value - h_intrinsic_out));
//	printf("Standard equals Intrinsic?\t%s diff=%e\n",
//		h_standard_out == h_intrinsic_out ? "Yes" : "No",
//		fabs(h_standard_out - h_intrinsic_out));
//	printf("\n");
//	printf("Mean execution time for standard function powf:    %f s\n",
//		mean_standard_time);
//	printf("Mean execution time for intrinsic function __powf: %f s\n",
//		mean_intrinsic_time);
//
//	return 0;
//}
