#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

//  to see the branch divergence we have to have conditions more than threshold (7 or 4 depend on complexity)
__global__ void mathkernel3(float * c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
	a = b = 0.0f;

	if (tid % 2 == 0)
	{
		a = 100.0f;
	}
	else if (tid % 3 == 0)
	{
		a = 200.0f;
	}
	else if (tid % 5 == 0)
	{
		a = 100;
	}
	else if (tid % 7 == 0)
	{
		a = 400.0f;
	}
	else if (tid % 9 == 0)
	{
		a = 500.0f;
	}
	else if (tid % 11 == 0)
	{
		a = 400;
	}
	else if (tid % 17 == 0)
	{
		a = 700.0f;
	}
	else if (tid % 19 == 0)
	{
		a = 300;
	}
	else if (tid % 23 == 0)
	{
		a = 200;
	}
	else if (tid % 29 == 0)
	{
		a = 1000.0f;
	}

	c[tid] = a + b;
}

__global__ void mathkernel2(float * c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
	a = b = 0.0f;

	bool ipred = (tid % 2 == 0);

	if (ipred)
	{
		a = 100.0f;
	}
	
	if(!ipred)
	{
		b = 200.0f;
	}
	c[tid] = a + b;
}

__global__ void mathkernel1(float * c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
	a = b = 0.0f;

	if (tid%2 ==0)
	{
		a = 100.0f;
	}
	else
	{
		b = 200.0f;
	}
	c[tid] = a + b;
}

void run_code_divergence()
{
	int array_size = 1024;
	int byte_array_size = array_size * sizeof(float);
	float * h_c, *h_ref;

	h_c = (float*)malloc(byte_array_size);
	h_ref = (float*)malloc(byte_array_size);

	for (int i = 0; i < array_size; i++)
	{
		h_c[i] = 1;
	}

	float* d_c;
	cudaMalloc((float**)&d_c,byte_array_size);
	cudaMemcpy(d_c, h_c, byte_array_size, cudaMemcpyHostToDevice);

	dim3 grid(1);
	dim3 block(array_size / grid.x);

	mathkernel1 << <grid, block >> > (d_c);
	cudaDeviceSynchronize();

	mathkernel2 << <grid, block >> > (d_c);
	cudaDeviceSynchronize();
	//cudaMemcpy(h_ref, d_c, byte_array_size, cudaMemcpyDeviceToHost);

	mathkernel3 << <grid, block >> > (d_c);
	cudaDeviceSynchronize();

	cudaFree(d_c);
	free(h_ref);
	free(h_c);
}

//int main()
//{
//	run_code_divergence();
//	system("pause");
//	return 0;
//}