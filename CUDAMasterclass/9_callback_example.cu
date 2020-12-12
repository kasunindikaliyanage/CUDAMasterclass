#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "cuda_common.cuh"

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data)
{
	printf("callback from stream %d\n", *((int *)data));
}

__global__ void cb_k1()
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid == 0)
	{
		printf("This is a test 1 \n");
	}
}

__global__ void cb_k2()
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid == 0)
	{
		printf("This is a test 2 \n");
	}
}

__global__ void cb_k3()
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid == 0)
	{
		printf("This is a test 3 \n");
	}
}

//int main(int argc, char ** argv)
//{
//	int size = 1 << 15;
//	cudaEvent_t event_str1;
//	gpuErrchk(cudaEventCreateWithFlags(&event_str1, cudaEventDisableTiming));
//
//	cudaStream_t stm1, stm2, stm3;
//	gpuErrchk(cudaStreamCreate(&stm1));
//	gpuErrchk(cudaStreamCreate(&stm2));
//	gpuErrchk(cudaStreamCreate(&stm3));
//
//	dim3 block(128);
//	dim3 grid(size / block.x);
//
//	cb_k1 << <grid, block, 0, stm1 >> > ();
//	cb_k2 << <grid, block, 0, stm2 >> > ();
//	cb_k3  << <grid, block, 0, stm3 >> > ();
//
//	gpuErrchk(cudaStreamDestroy(stm1));
//	gpuErrchk(cudaStreamDestroy(stm2));
//	gpuErrchk(cudaStreamDestroy(stm3));
//	gpuErrchk(cudaDeviceSynchronize());
//
//	gpuErrchk(cudaDeviceReset());
//	return 0;
//}
