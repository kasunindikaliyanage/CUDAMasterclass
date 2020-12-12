 #include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "common.h"

__global__ void sumArrays(int *A, int *B, int *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(int *A, int *B, int *C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N) C[i] = A[i] + B[i];
}

//int main(int argc, char **argv)
//{
//	// set up device
//	int dev = 0;
//	cudaSetDevice(dev);
//
//	// get device properties
//	cudaDeviceProp deviceProp;
//	cudaGetDeviceProperties(&deviceProp, dev);
//
//	// check if support mapped memory
//	if (!deviceProp.canMapHostMemory)
//	{
//		printf("Device %d does not support mapping CPU host memory!\n", dev);
//		cudaDeviceReset();
//		exit(EXIT_SUCCESS);
//	}
//
//	// set up data size of vectors
//	int power = 22;
//
//	if (argc > 1) power = atoi(argv[1]);
//
//	int nElem = 1 << power;
//	size_t nBytes = nElem * sizeof(int);
//
//	// part 1: using device memory
//	// malloc host memory
//	int *h_A, *h_B, *hostRef, *gpuRef;
//	h_A = (int *)malloc(nBytes);
//	h_B = (int *)malloc(nBytes);
//	hostRef = (int *)malloc(nBytes);
//	gpuRef = (int *)malloc(nBytes);
//
//	// initialize data at host side
//	initialize(h_A, nElem,INIT_ONE_TO_TEN);
//	initialize(h_B, nElem);
//	memset(gpuRef, 0, nBytes);
//
//	// malloc device global memory
//	int *d_A, *d_B, *d_C;
//	cudaMalloc((int**)&d_A, nBytes);
//	cudaMalloc((int**)&d_B, nBytes);
//	cudaMalloc((int**)&d_C, nBytes);
//
//	// transfer data from host to device
//	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
//
//	// set up execution configuration
//	int iLen = 512;
//	dim3 block(iLen);
//	dim3 grid((nElem + block.x - 1) / block.x);
//
//	sumArrays << <grid, block >> >(d_A, d_B, d_C, nElem);
//	
//	cudaDeviceSynchronize();
//	// copy kernel result back to host side
//	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
//
//	// free device global memory
//	cudaFree(d_A);
//	cudaFree(d_B);
//
//	// free host memory
//	free(h_A);
//	free(h_B);
//
//	// part 2: using zerocopy memory for array A and B
//	// allocate zerocpy memory
//	cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped);
//	cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped);
//
//	// initialize data at host side
//	initialize(h_A, nElem, INIT_ONE_TO_TEN);
//	initialize(h_B, nElem, INIT_ONE_TO_TEN);
//	memset(gpuRef, 0, nBytes);
//
//	// get the mapped device pointer
//	cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
//	cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);
//
//	// execute kernel with zero copy memory
//	sumArraysZeroCopy << <grid, block >> >(d_A, d_B, d_C, nElem);
//
//	cudaDeviceSynchronize();
//
//	// copy kernel result back to host side
//	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
//
//	// free  memory
//	cudaFree(d_C);
//	cudaFreeHost(h_A);
//	cudaFreeHost(h_B);
//
//	free(hostRef);
//	free(gpuRef);
//
//	// reset device
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//}
