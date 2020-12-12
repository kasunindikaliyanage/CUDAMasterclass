#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>


// Matrix transpose kernel - Consider 1 block contain all the elements in 1 row of the Matrix
//                           So 1024 is the maximum number or elements should contain in a row
//				 IMPORTANT : Assuming grid is 1D and nx=ny
//
// param 1 (input)  - input matrix in 1D array
// param 2 (output) - tranpose of the input matrix in 1D array
// param 3 (nx)     - number of elements in input matrix's row 
// param 4 (ny)     - number of elements in input matrix's column
__global__ void matrix_transpose_k1(float* input,float* output,const int nx, const int ny)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = threadIdx.x*blockDim.x;
	//printf("gid : %d , offset : %d , index : %d ,value : %f \n", gid, offset, offset + blockIdx.x,input[offset + blockIdx.x]);
	output[gid] = input[offset + blockIdx.x];
}

// Copy the input matrix elements to output matrix with coalesed access and write it same way
// param 1 (input)  - input matrix in 1D array
// param 2 (output) - tranpose of the input matrix in 1D array
// param 3 (nx)     - number of elements in input matrix's row 
// param 4 (ny)     - number of elements in input matrix's column
__global__ void copy_rows(float* input, float* output, const int nx, const int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		output[iy*ny + ix] = input[iy*nx + ix];
	}
}

//// Copy the input matrix elements to output matrix with stride and write it same way
// param 1 (input)  - input matrix in 1D array
// param 2 (output) - tranpose of the input matrix in 1D array
// param 3 (nx)     - number of elements in input matrix's row 
// param 4 (ny)     - number of elements in input matrix's column
__global__ void copy_columns(float* input, float* output, const int nx, const int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (ix < nx && iy < ny)
	{
		output[ix*ny + iy] = input[ix*ny + iy];
	}
}

// Read the elements from input matrix in coalesed manner and write to output matrix in stride manner
// param 1 (input)  - input matrix in 1D array
// param 2 (output) - tranpose of the input matrix in 1D array
// param 3 (nx)     - number of elements in input matrix's row 
// param 4 (ny)     - number of elements in input matrix's column
__global__ void read_coaleased_write_stride_mat_trans(float* input, float* output, const int nx, const int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		output[ix*ny + iy] = input[iy*nx + ix];
	}
}

__global__ void read_stride_write_coaleased_mat_trans(float* input, float* output, const int nx, const int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		output[iy*nx + ix] = input[ix*ny + iy];
	}
}

void run_matrix_transpose_k1()
{
	int ny = 1<<15;
	int nx = 1 << 15;

	int blockx = 128;
	int blocky = 128;

	float * h_matrix, *h_output;

	int mat_size = nx * ny;
	int mat_byte_size = sizeof(int)*mat_size;

	h_matrix = (float*)malloc(mat_byte_size);
	h_output = (float*)malloc(mat_byte_size);

	for (int i = 0; i < mat_size; i++)
	{
		h_matrix[i] = i;
	}

	printf("Printing input matrix \n");
	//for (int i = 0; i < mat_size; i+=32)
	//{
	//	if (i != 0 && i%nx == 0)
	//	{
	//		printf("\n");
	//	}
	//	printf(" %1.0f ", h_matrix[i]);
	//}
	//printf("\n");

	dim3 grid(blockx,blocky);
	dim3 blocks((nx+blockx-1)/blockx, (ny + blocky - 1) / blocky);

	float * d_matrix, *d_output;

	cudaMalloc((float**)&d_matrix,mat_byte_size);
	cudaMalloc((float**)&d_output, mat_byte_size);

	cudaMemcpy(d_matrix,h_matrix,mat_byte_size,cudaMemcpyHostToDevice);
	copy_rows << <grid,blocks >> > (d_matrix,d_output,nx,ny);
	cudaDeviceSynchronize();

	cudaMemcpy(h_output,d_output,mat_byte_size,cudaMemcpyDeviceToHost);

	printf("Printing output matrix \n");
	/*
	for (int i = 0; i < ny; i++)
	{
		if (i != 0 && i%ny == 0)
		{
			printf("\n");
		}
		printf(" %1.0f ",h_output[i]);
	}
	printf("\n");*/
}


//int main()
//{
//	run_matrix_transpose_k1();
//	system("pause");
//	return 0;
//}
