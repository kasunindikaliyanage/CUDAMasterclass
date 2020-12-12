#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void global_mem_limit_test(int * input,int size)
{
	int gid = blockDim.x*blockIdx.x + threadIdx.x;
	//printf(" gid : %d \n", gid);

	//if (gid < size)
	//	input[gid] = gid;

	//if (gid == (size - 1))
	//	printf("final element value : %d, gid : %d \n",input[gid],gid);
}

void run_mem_limit_test(int argc, char ** argv)
{
	int in_size = 1 << 22;
	int block_size = 1 << 10;
	long long in_byte_size = in_size * sizeof(int);
	float in_MB = (in_byte_size / (1024 * 1024));

	printf("Application is going to transfer %4.2f MB \n",in_MB);

	if (argc > 1)
		in_size = 1 << (atoi(argv[1]));

	//declare and initialize input array
	int * h_in;
	h_in = (int*)malloc(in_byte_size);
	if (h_in == NULL)
		printf("Insufficient memory available\n");

	memset(h_in, 0, in_byte_size);

	int * d_in;

	dim3 grid((in_size)/block_size);
	dim3 block(block_size);

	cudaError error;

	error =cudaMalloc((void**)&d_in,in_byte_size);
	if (error)
	{
		printf("Error : %s", cudaGetErrorString(cudaGetLastError()));
	}

	error = cudaMemcpy(d_in,h_in,in_byte_size,cudaMemcpyHostToDevice);
	if (error)
	{
		printf("Error : %s", cudaGetErrorString(cudaGetLastError()));
	}

	global_mem_limit_test << <grid, block >> > (d_in, in_size);
	
	error =cudaDeviceSynchronize();
	if (error)
	{
		printf("Error : %s", cudaGetErrorString(cudaGetLastError()));
	}

	error =cudaFree(d_in);
	if (error)
	{
		printf("Error : %s", cudaGetErrorString(cudaGetLastError()));
	}

	cudaFree(d_in);
	free(h_in);
}

//int main(int argc, char ** argv)
//{
//	run_mem_limit_test(argc, argv);
//	return 0;
//}