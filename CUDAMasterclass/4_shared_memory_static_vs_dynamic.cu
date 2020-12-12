#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#define BDIMX 32
#define BDIMY 32

__global__ void setRowReadColDyn(int * out)
{
	extern __shared__ int tile[];

	int row_index = threadIdx.y * blockDim.x + threadIdx.x;
	int col_index = threadIdx.x * blockDim.y + threadIdx.y;

	tile[row_index] = row_index;

	__syncthreads();

	out[row_index] = tile[col_index];

}

//int main(int argc, char **argv)
//{
//	cudaSharedMemConfig pConfig;
//	cudaDeviceGetSharedMemConfig(&pConfig);
//	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");
//
//	// set up array size 2048
//	int nx = BDIMX;
//	int ny = BDIMY;
//
//	bool iprintf = 0;
//
//	if (argc > 1) iprintf = atoi(argv[1]);
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
//	setRowReadColDyn << <grid, block, sizeof(int) * (nx*ny) >> >(d_C);
//	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
//
//	// free host and device memory
//	cudaFree(d_C);
//	free(gpuRef);
//
//	// reset device
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//}
