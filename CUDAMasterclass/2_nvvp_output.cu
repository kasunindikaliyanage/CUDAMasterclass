#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

__global__ void stream_test(int* in, int * out, int size)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid < size)
	{
		//THIS FOR LOOP IS ONLY FOR VISUALIZING PURPOSE  
		for (int  i = 0; i < 25; i++)
		{
			out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
		}
	}
}

//int main(int argc, char ** argv)
//{
//	int size = 1 << 18;
//	int byte_size = size * sizeof(int);
//
//	//initialize host pointer
//	int* h_in, *h_ref;
//	h_in = (int *)malloc(byte_size);
//	h_ref = (int *)malloc(byte_size);
//	initialize(h_in,INIT_RANDOM);
//
//	//allocate device pointers
//	int * d_in, *d_out;
//	cudaMalloc((void**)&d_in, byte_size);
//	cudaMalloc((void**)&d_out, byte_size);
//
//	//transfer data from host to device
//	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
//	
//	//kernel launch
//	dim3 block(128);
//	dim3 grid(size / block.x);
//	
//	stream_test << <grid, block >>> (d_in,d_out, size);
//	cudaDeviceSynchronize();
//
//	//copy the memory back to host
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	cudaDeviceReset();
//	return 0;
//}