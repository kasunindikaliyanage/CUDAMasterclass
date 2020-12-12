#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "cuda_common.cuh"

__global__ void blocking_nonblocking_test1()
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid == 0)
	{
		for (size_t i = 0; i < 10000; i++)
		{
			printf("kernel 1 \n");
		}
	}
}

//int main(int argc, char ** argv)
//{
//	int size = 1 << 15;
//	
//	cudaStream_t stm1,stm2,stm3;
//	gpuErrchk(cudaStreamCreateWithFlags(&stm1, cudaStreamNonBlocking));
//	gpuErrchk(cudaStreamCreate(&stm2));
//	gpuErrchk(cudaStreamCreateWithFlags(&stm3,cudaStreamNonBlocking));
//
//
//	dim3 block(128);
//	dim3 grid(size / block.x);
//
//	blocking_nonblocking_test1 << <grid, block, 0 , stm1 >> > ();
//	blocking_nonblocking_test1 << <grid, block >> > ();
//	blocking_nonblocking_test1 << <grid, block, 0, stm3 >> > ();
//
//	gpuErrchk(cudaStreamDestroy(stm1));
//	gpuErrchk(cudaStreamDestroy(stm2));
//	gpuErrchk(cudaStreamDestroy(stm3));
//	gpuErrchk(cudaDeviceSynchronize());
//
//	gpuErrchk(cudaDeviceReset());
//	return 0;
//}