#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define LEN 1<<22

struct testStruct {
	int x;
	int y;
};

struct structArray
{
	int x[LEN];
	int y[LEN];
};

__global__ void test_aos(testStruct * in, testStruct * result, const int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		testStruct temp = in[gid];
		temp.x += 5;
		temp.y += 10;
		result[gid] = temp;
	}
}

__global__ void test_soa(structArray *data, structArray * result, const int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		float tmpx = data->x[gid];
		float tmpy = data->y[gid];

		tmpx += 5;
		tmpy += 10;
		result->x[gid] = tmpx;
		result->y[gid] = tmpy;
	}
}

void testAOS()
{
	printf(" testing AOS \n");

	int array_size = LEN;
	int byte_size = sizeof(testStruct)* array_size;
	int block_size = 128;

	testStruct * h_in, *h_ref;
	h_in = (testStruct*)malloc(byte_size);
	h_ref = (testStruct*)malloc(byte_size);

	for (int i = 0; i < array_size; i++)
	{
		h_in[i].x = 1;
		h_in[i].y = 2;
	}

	testStruct * d_in, *d_results;
	cudaMalloc((void**)&d_in, byte_size);
	cudaMalloc((void**)&d_results, byte_size);

	cudaMemcpy(d_in,h_in,byte_size, cudaMemcpyHostToDevice);

	dim3 block(block_size);
	dim3 grid(array_size / (block.x));

	test_aos << <grid, block >> > (d_in, d_results, array_size);

	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_results, byte_size, cudaMemcpyDeviceToHost);

	cudaFree(d_results);
	cudaFree(d_in);
	free(h_ref);
	free(h_in);

	cudaDeviceReset();
}

void testSOA()
{
	printf(" testing SOA \n");

	int array_size = LEN;
	int byte_size = sizeof(structArray);
	int block_size = 128;

	structArray *h_in = (structArray*)malloc(byte_size);
	structArray *h_ref = (structArray*)malloc(byte_size);

	for (int i = 0; i < array_size; i++)
	{
		h_in->x[i] = 1;
		h_in->y[i] = 2;
	}

	structArray* d_in, *d_results;
	cudaMalloc((structArray**)&d_in, byte_size);
	cudaMalloc((structArray**)&d_results, byte_size);

	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);

	dim3 block(block_size);
	dim3 grid(array_size / (block.x));

	test_soa << <grid, block >> > (d_in, d_results, array_size);

	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_results, byte_size, cudaMemcpyDeviceToHost);

	cudaFree(d_results);
	cudaFree(d_in);
	free(h_ref);
	free(h_in);

	cudaDeviceReset();
}

//int main(int argc, char** argv)
//{
//	int kernel_ind = 0;
//
//	if (argc > 1)
//	{
//		kernel_ind = atoi(argv[1]);
//	}
//
//	if (kernel_ind == 0)
//	{
//		testAOS();
//	}
//	else
//	{
//		testSOA();
//	}
//
//	return EXIT_SUCCESS;
//}
