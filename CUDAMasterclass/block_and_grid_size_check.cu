#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

void initialData_1(float *in, const int size)
{
	for (int i = 0; i < size; i++)
	{
		in[i] = (float)(rand() & 0xFF) / 10.0f; //100.0f;
	}
	return;
}


// case 0 copy kernel: access data in rows
__global__ void copyRow_1(float *out, float *in, const int nx, const int ny)
{
	unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		out[iy * nx + ix] = in[iy * nx + ix];
	}
}


// main functions
//int main(int argc, char **argv)
//{
//	// set up device
//	int dev = 0;
//	cudaDeviceProp deviceProp;
//	cudaError error;
//
//	error = cudaGetDeviceProperties(&deviceProp, dev);
//	printf("%s starting transpose at ", argv[0]);
//	printf("device %d: %s ", dev, deviceProp.name);
//	printf("allowed memory size : %d",(int)deviceProp.totalGlobalMem);
//	error = cudaSetDevice(dev);
//
//	// set up array size 2048
//	int nx = 1 << 5;
//	int ny = 1 << 5;
//
//	// select a kernel and block size
//	int iKernel = 1;
//	int blockx = 16;
//	int blocky = 16;
//
//	if (argc > 1) blockx = atoi(argv[1]);
//
//	if (argc > 2) blocky = atoi(argv[2]);
//
//	if (argc > 3) nx = atoi(argv[3]);
//
//	if (argc > 4) ny = atoi(argv[4]);
//
//	
//	size_t nBytes = nx * ny * sizeof(float);
//	float bytes = nBytes / (1024 * 1024);
//	printf(" with matrix nx %d ny %d with kernel %d with %.2f MB memory\n", nx, ny, iKernel, bytes);
//
//	// execution configuration
//	dim3 block(blockx, blocky);
//	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
//
//	// allocate host memory
//	float *h_A = (float *)malloc(nBytes);
//	float *hostRef = (float *)malloc(nBytes);
//	float *gpuRef = (float *)malloc(nBytes);
//
//	// initialize host array
//	initialData_1(h_A, nx * ny);
//
//	// allocate device memory
//	float *d_A, *d_C;
//	error = cudaMalloc((float**)&d_A, nBytes);
//	error = cudaMalloc((float**)&d_C, nBytes);
//
//	// copy data from host to device
//	error = cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
//	error = cudaGetLastError();
//	printf("%s - %s \n", cudaGetErrorName(error),cudaGetErrorString(error));
//
//	copyRow_1 << <grid, block >> >(d_C, d_A, nx, ny);
//	error = cudaDeviceSynchronize();
//	
//	printf("%s <<< grid (%d,%d) block (%d,%d)>>> ", "CopyRow", grid.x, grid.y, block.x,
//		block.y);
//	cudaGetLastError();
//
//	// check kernel results
//	if (iKernel > 1)
//	{
//		error = cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
//	}
//
//	// free host and device memory
//	error = cudaFree(d_A);
//	error = cudaFree(d_C);
//	free(h_A);
//	free(hostRef);
//	free(gpuRef);
//
//	// reset device
//	error = cudaDeviceReset();
//	system("pause");
//	return EXIT_SUCCESS;
//}
