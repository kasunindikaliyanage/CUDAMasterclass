#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#define BDIMX 32
#define BDIMY 32

__global__ void setRowReadCol(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];

	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	//store to the shared memory
	tile[threadIdx.y][threadIdx.x] = idx;

	//waiting for all the threads in thread block to reach this point
	__syncthreads();

	//load from shared memory
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRow(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];

	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	//store to the shared memory
	tile[threadIdx.x][threadIdx.y] = idx;

	//waiting for all the threads in thread block to reach this point
	__syncthreads();

	//load from shared memory
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRow(int * out)
{
	__shared__ int tile[BDIMY][BDIMX];

	int idx = threadIdx.y * blockDim.x + threadIdx.x;

	//store to the shared memory
	tile[threadIdx.y][threadIdx.x] = idx;

	//waiting for all the threads in thread block to reach this point
	__syncthreads();

	//load from shared memory
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

//int main(int argc, char **argv)
//{
//	int memconfig = 0;
//	if (argc > 1)
//	{
//		memconfig = atoi(argv[1]);
//	}
//
//
//	if (memconfig == 1)
//	{
//		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
//	}
//	else
//	{
//		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
//	}
//
//	
//	cudaSharedMemConfig pConfig;
//	cudaDeviceGetSharedMemConfig(&pConfig);
//	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
//	
//
//	// set up array size 2048
//	int nx = BDIMX;
//	int ny = BDIMY;
//
//	bool iprintf = 0;
//	
//	if (argc > 2) iprintf = atoi(argv[1]);
//
//	size_t nBytes = nx * ny * sizeof(int);
//
//	// execution configuration
//	dim3 block(BDIMX, BDIMY);
//	dim3 grid(1, 1);
//	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
//		block.y);
//
//	// allocate device memory
//	int *d_C;
//	cudaMalloc((int**)&d_C, nBytes);
//	int *gpuRef = (int *)malloc(nBytes);
//
//	cudaMemset(d_C, 0, nBytes);
//	setColReadRow << <grid, block >> >(d_C);
//	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
//
//	if (iprintf)  printData("set col read col   ", gpuRef, nx * ny);
//
//	cudaMemset(d_C, 0, nBytes);
//	setRowReadRow << <grid, block >> >(d_C);
//	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
//
//	if (iprintf)  printData("set row read row   ", gpuRef, nx * ny);
//
//	cudaMemset(d_C, 0, nBytes);
//	setRowReadCol << <grid, block >> >(d_C);
//	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
//
//	if (iprintf)  printData("set row read col   ", gpuRef, nx * ny);
//
//	// free host and device memory
//	cudaFree(d_C);
//	free(gpuRef);
//
//	// reset device
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//}
