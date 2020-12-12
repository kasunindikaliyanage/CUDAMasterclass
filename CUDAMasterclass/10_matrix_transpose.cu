#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

__global__ void copy_row(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[iy * nx + ix] = mat[iy * nx + ix];
	}
}

__global__ void copy_column(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[ix * ny + iy];
	}
}

__global__ void transpose_read_row_write_column(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}

__global__ void transpose_read_column_write_row(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[iy * nx + ix] = mat[ix * ny + iy];
	}
}

__global__ void transpose_unroll4_row(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int ti = iy * nx + ix;
	int to = ix * ny + iy;

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		transpose[to]						= mat[ti];
		transpose[to + ny*blockDim.x]		= mat[ti + blockDim.x];
		transpose[to + ny * 2 * blockDim.x] = mat[ti + 2 * blockDim.x];
		transpose[to + ny * 3 * blockDim.x] = mat[ti + 3 * blockDim.x];
	}
}

__global__ void transpose_unroll4_col(int * mat, int * transpose, int nx, int ny)
{
	int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	int ti = iy * nx + ix;
	int to = ix * ny + iy;

	if (ix + 3 * blockDim.x < nx && iy < ny)
	{
		transpose[ti] = mat[to];
		transpose[ti + blockDim.x] = mat[to + blockDim.x*ny];
		transpose[ti + 2 * blockDim.x] = mat[to + 2 * blockDim.x*ny];
		transpose[ti + 3 * blockDim.x] = mat[to + 3 * blockDim.x*ny];
	}
}

__global__ void transpose_diagonal_row(int * mat, int * transpose, int nx, int ny)
{
	int blk_x = blockIdx.x;
	int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;

	int ix = blockIdx.x * blk_x + threadIdx.x;
	int iy = blockIdx.y * blk_y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}


//int main(int argc, char** argv)
//{
//	//default values for variabless
//	int nx = 1024;
//	int ny = 1024;
//	int block_x = 128;
//	int block_y = 8;
//	int kernel_num = 0;
//
//	if (argc > 1)
//		kernel_num = atoi(argv[1]);
//
//	int size = nx * ny;
//	int byte_size = sizeof(int*) * size;
//
//	printf("Matrix transpose for %d X % d matrix with block size %d X %d \n",nx,ny,block_x,block_y);
//
//	int * h_mat_array = (int*)malloc(byte_size);
//	int * h_trans_array = (int*)malloc(byte_size);
//	int * h_ref = (int*)malloc(byte_size);
//
//	//initialize matrix with integers between one and ten
//	initialize(h_mat_array,size ,INIT_ONE_TO_TEN);
//
//	//matirx transpose in CPU
//	mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);
//
//	int * d_mat_array, *d_trans_array;
//	
//	cudaMalloc((void**)&d_mat_array, byte_size);
//	cudaMalloc((void**)&d_trans_array, byte_size);
//
//	cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);
//
//	dim3 blocks(block_x, block_y);
//	dim3 grid(nx/block_x, ny/block_y);
//
//	void(*kernel)(int*, int*, int, int);
//	char * kernel_name;
//
//	switch (kernel_num)
//	{
//	case 0:
//		kernel = &copy_row;
//		kernel_name = "Copy row   ";
//		break;
//	case 1 :
//		kernel = &copy_column;
//		kernel_name = "Copy column   ";
//		break;
//	case 2 :
//		kernel = &transpose_read_row_write_column;
//		kernel_name = " Read row write column ";
//		break;
//	case 3:
//		kernel = &transpose_read_column_write_row;
//		kernel_name = "Read column write row ";
//		break;
//	case 4:
//		kernel = &transpose_unroll4_row;
//		kernel_name = "Unroll 4 row ";
//		break;
//	case 5:
//		kernel = &transpose_unroll4_col;
//		kernel_name = "Unroll 4 col ";
//		break;
//	case 6:
//		kernel = &transpose_diagonal_row;
//		kernel_name = "Diagonal row ";
//		break;
//	}
//
//	printf(" Launching kernel %s \n",kernel_name);
//
//	clock_t gpu_start, gpu_end;
//	gpu_start = clock();
//
//	kernel <<< grid, blocks>> > (d_mat_array, d_trans_array,nx, ny);
//
//	cudaDeviceSynchronize();
//
//	gpu_end = clock();
//	print_time_using_host_clock(gpu_start, gpu_end);
//
//	//copy the transpose memroy back to cpu
//	cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost);
//
//	//compare the CPU and GPU transpose matrix for validity
//	compare_arrays(h_ref, h_trans_array, size);
//
//	cudaDeviceReset();
//	return EXIT_SUCCESS;
//}