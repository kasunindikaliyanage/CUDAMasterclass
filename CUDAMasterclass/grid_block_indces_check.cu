#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void check_index()
{
	printf(" threadidx:(%d,%d,%d) blockidx:(%d,%d,%d) blockdim:(%d,%d,%d) griddim:(%d,%d,%d) \n",
	threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,
		blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);
}

//int main()
//{
//	int element_count = 6;
//
//	dim3 block(3);
//	dim3 grid((element_count+block.x -1)/block.x);
//
//	printf("grid.x:%d grid.y:%d grid.z:%d \n",grid.x,grid.y,grid.z);
//	printf("block.x:%d block.y:%d block.z:%d \n", block.x, block.y, block.z);
//
//	check_index << <grid,block >> > ();
//	cudaDeviceReset();
//
//	system("pause");
//	return 0;
//}