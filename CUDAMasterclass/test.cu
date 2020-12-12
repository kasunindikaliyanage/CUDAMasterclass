//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//
//#include "common.h"
//#include "cuda_common.cuh"
//
//#include "cuda.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#define c0 1
//#define c1 2
//#define c2 3
//#define c3 4
//#define c4 5
//
//#define RADIUS 4
//
//#define BDIM 128
//
////constant memory declaration
//__constant__ int coef[9];
//
//void host_const_calculation(int * in, int * out, int size)
//{
//	for (int i = 0; i < size; i++)
//	{
//
//		if (i < RADIUS)
//		{
//			out[i] = in[i + 4] * c0
//				+ in[i + 3] * c1
//				+ in[i + 2] * c2
//				+ in[i + 1] * c3
//				+ in[i] * c4;
//
//			if (i == 3)
//			{
//				out[i] += in[2] * c3;
//				out[i] += in[1] * c2;
//				out[i] += in[0] * c1;
//			}
//			else if (i == 2)
//			{
//				out[i] += in[1] * c3;
//				out[i] += in[0] * c2;
//			}
//			else if (i == 1)
//			{
//				out[i] += in[0] * c3;
//			}
//		}
//		else if ((i + RADIUS) >= size)
//		{
//			out[i] = in[i - 4] * c0
//				+ in[i - 3] * c1
//				+ in[i - 2] * c2
//				+ in[i - 1] * c3
//				+ in[i] * c4;
//
//			if (i == size - 4)
//			{
//				out[i] += in[size - 3] * c3;
//				out[i] += in[size - 2] * c2;
//				out[i] += in[size - 1] * c1;
//			}
//			else if (i == size - 3)
//			{
//				out[i] += in[size - 2] * c3;
//				out[i] += in[size - 1] * c2;
//			}
//			else if (i == size - 2)
//			{
//				out[i] += in[size - 1] * c3;
//			}
//		}
//		else
//		{
//			out[i] = (in[i - 4] + in[i + 4])*c0
//				+ (in[i - 3] + in[i + 3])*c1
//				+ (in[i - 2] + in[i + 2])*c2
//				+ (in[i - 1] + in[i + 1])*c3
//				+ in[i] * c4;
//		}
//	}
//}
//
////setting up constant memory from host
//void setup_coef_1()
//{
//	const int h_coef[] = { c0,c1,c2,c3,c4,c3,c2,c1,c0 };
//	cudaMemcpyToSymbol(coef, h_coef, (9) * sizeof(float));
//}
//
//__global__ void constant_stencil_smem_test(int * in,
//	int *out, int size)
//{
//	__shared__ int smem[BDIM + 2 * RADIUS];
//
//	int gid = blockDim.x * blockIdx.x + threadIdx.x;
//	int bid = blockIdx.x;
//	int num_of_blocks = gridDim.x;
//
//	int value = 0;
//
//	if (gid < size)
//	{
//		int sidx = threadIdx.x + RADIUS;
//
//		// Read data from global memory into shared memory
//		smem[sidx] = in[gid];
//
//		if (bid != 0 && bid != (num_of_blocks - 1))
//		{
//			if (threadIdx.x < RADIUS)
//			{
//				smem[sidx - RADIUS] = in[gid - RADIUS];
//				smem[sidx + BDIM] = in[gid + BDIM];
//			}
//		}
//		else if (bid == 0)
//		{
//			if (threadIdx.x < RADIUS)
//			{
//				smem[sidx - RADIUS] = 0;
//				smem[sidx + BDIM] = in[gid + BDIM];
//			}
//		}
//		else
//		{
//			if (threadIdx.x < RADIUS)
//			{
//				smem[sidx - RADIUS] = in[gid - RADIUS];
//				smem[sidx + BDIM] = 0;
//			}
//		}
//
//		__syncthreads();
//
//		value += smem[sidx - 4] * coef[0];
//		value += smem[sidx - 3] * coef[1];
//		value += smem[sidx - 2] * coef[2];
//		value += smem[sidx - 1] * coef[3];
//		value += smem[sidx - 0] * coef[4];
//		value += smem[sidx + 1] * coef[5];
//		value += smem[sidx + 2] * coef[6];
//		value += smem[sidx + 3] * coef[7];
//		value += smem[sidx + 4] * coef[8];
//
//		out[gid] = value;
//	}
//}
//
//
//int main(int argc, char ** argv)
//{
//	int size = 1 << 22;
//	int byte_size = sizeof(int) * size;
//	int block_size = BDIM;
//
//	int * h_in, *h_out, *h_ref;
//
//	h_in = (int*)malloc(byte_size);
//	h_out = (int*)malloc(byte_size);
//	h_ref = (int*)malloc(byte_size);
//
//	initialize(h_in, size, INIT_ONE);
//
//	int * d_in, *d_out;
//	cudaMalloc((void**)&d_in, byte_size);
//	cudaMalloc((void**)&d_out, byte_size);
//
//	cudaMemcpy(d_in, h_in, byte_size, cudaMemcpyHostToDevice);
//	cudaMemset(d_out, 0, byte_size);
//
//	setup_coef_1();
//
//	dim3 blocks(block_size);
//	dim3 grid(size / blocks.x);
//
//	constant_stencil_smem_test << < grid, blocks >> > (d_in, d_out, size);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost);
//
//	host_const_calculation(h_in, h_out, size);
//
//	compare_arrays(h_ref, h_out, size);
//
//	cudaFree(d_out);
//	cudaFree(d_in);
//	free(h_ref);
//	free(h_out);
//	free(h_in);
//
//	return 0;
//}//#include <stdio.h>
////#include <stdlib.h>
////#include <time.h>
////
////#include "common.h"
////#include "cuda_common.cuh"
////
////#include "cuda.h"
////#include "cuda_runtime.h"
////#include "device_launch_parameters.h"
////
////#define BDIMX 64
////#define BDIMY 8
////#define IPAD 2
////
////__global__ void transpose_read_raw_write_column_benchmark(int * mat,
////	int* transpose, int nx, int ny)
////{
////	int ix = blockDim.x * blockIdx.x + threadIdx.x;
////	int iy = blockDim.y * blockIdx.y + threadIdx.y;
////
////	if (ix < nx && iy < ny)
////	{
////		//read by row, write by col
////		transpose[ix * ny + iy] = mat[iy * nx + ix];
////	}
////}
////
////__global__ void transpose_smem(int * in, int* out, int nx, int ny)
////{
////	__shared__ int tile[BDIMY][BDIMX];
////
////	//input index
////	int ix, iy, in_index;
////
////	//output index
////	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;
////
////	//ix and iy calculation for input index
////	ix = blockDim.x * blockIdx.x + threadIdx.x;
////	iy = blockDim.y * blockIdx.y + threadIdx.y;
////
////	//input index
////	in_index = iy * nx + ix;
////
////	//1D index calculation fro shared memory
////	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;
////
////	//col major row and col index calcuation
////	i_row = _1d_index / blockDim.y;
////	i_col = _1d_index % blockDim.y;
////
////	//coordinate for transpose matrix
////	out_ix = blockIdx.y * blockDim.y + i_col;
////	out_iy = blockIdx.x * blockDim.x + i_row;
////
////	//output array access in row major format
////	out_index = out_iy * ny + out_ix;
////
////	if (ix < nx && iy < ny)
////	{
////		//load from in array in row major and store to shared memory in row major
////		tile[threadIdx.y][threadIdx.x] = in[in_index];
////
////		//wait untill all the threads load the values
////		__syncthreads();
////
////		out[out_index] = tile[i_col][i_row];
////	}
////}
////
////__global__ void transpose_smem_pad(int * in, int* out, int nx, int ny)
////{
////	__shared__ int tile[BDIMY][BDIMX + IPAD];
////
////	//input index
////	int ix, iy, in_index;
////
////	//output index
////	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;
////
////	//ix and iy calculation for input index
////	ix = blockDim.x * blockIdx.x + threadIdx.x;
////	iy = blockDim.y * blockIdx.y + threadIdx.y;
////
////	//input index
////	in_index = iy * nx + ix;
////
////	//1D index calculation fro shared memory
////	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;
////
////	//col major row and col index calcuation
////	i_row = _1d_index / blockDim.y;
////	i_col = _1d_index % blockDim.y;
////
////	//coordinate for transpose matrix
////	out_ix = blockIdx.y * blockDim.y + i_col;
////	out_iy = blockIdx.x * blockDim.x + i_row;
////
////	//output array access in row major format
////	out_index = out_iy * ny + out_ix;
////
////	if (ix < nx && iy < ny)
////	{
////		//load from in array in row major and store to shared memory in row major
////		tile[threadIdx.y][threadIdx.x] = in[in_index];
////
////		//wait untill all the threads load the values
////		__syncthreads();
////
////		out[out_index] = tile[i_col][i_row];
////	}
////}
////
////__global__ void transpose_smem_pad_unrolling(int * in, int* out, int nx, int ny)
////{
////	__shared__ int tile[BDIMY*(2 * BDIMX + IPAD)];
////
////	//input index
////	int ix, iy, in_index;
////
////	//output index
////	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;
////
////	//ix and iy calculation for input index
////	ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
////	iy = blockDim.y * blockIdx.y + threadIdx.y;
////
////	//input index
////	in_index = iy * nx + ix;
////
////	//1D index calculation fro shared memory
////	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;
////
////	//col major row and col index calcuation
////	i_row = _1d_index / blockDim.y;
////	i_col = _1d_index % blockDim.y;
////
////	//coordinate for transpose matrix
////	out_ix = blockIdx.y * blockDim.y + i_col;
////	out_iy = 2 * blockIdx.x * blockDim.x + i_row;
////
////	//output array access in row major format
////	out_index = out_iy * ny + out_ix;
////
////	if (ix < nx && iy < ny)
////	{
////		unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
////		tile[row_idx] = in[in_index];
////		tile[row_idx + BDIMX] = in[in_index + BDIMX];
////
////		// thread synchronization   
////		__syncthreads();
////
////		// store two rows to global memory from two columns of shared memory    
////		unsigned int col_idx = i_col*(blockDim.x * 2 + IPAD) + i_row;
////		out[out_index] = tile[col_idx];
////		out[out_index + ny*BDIMX] = tile[col_idx + BDIMX];
////	}
////}
////
////int main(int argc, char** argv)
////{
////	//default values for variabless
////	int nx = 1024;
////	int ny = 1024;
////	int block_x = BDIMX;
////	int block_y = BDIMY;
////	int kernel_num = 0;
////
////	//set the variable based on arguments
////	if (argc > 1)
////		nx = 1 << atoi(argv[1]);
////	if (argc > 2)
////		ny = 1 << atoi(argv[2]);
////	if (argc > 3)
////		block_x = 1 << atoi(argv[3]);
////	if (argc > 4)
////		block_y = 1 << atoi(argv[4]);
////
////	int size = nx * ny;
////	int byte_size = sizeof(int*) * size;
////
////	printf("Matrix transpose for %d X % d matrix with block size %d X %d \n", nx, ny, block_x, block_y);
////
////	int * h_mat_array = (int*)malloc(byte_size);
////	int * h_trans_array = (int*)malloc(byte_size);
////	int * h_ref = (int*)malloc(byte_size);
////
////	initialize(h_mat_array, size, INIT_ONE_TO_TEN);
////
////	//matirx transpose in CPU
////	mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);
////
////	int * d_mat_array, *d_trans_array;
////
////	gpuErrchk(cudaMalloc((void**)&d_mat_array, byte_size));
////	gpuErrchk(cudaMalloc((void**)&d_trans_array, byte_size));
////
////	gpuErrchk(cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice));
////	gpuErrchk(cudaMemset(d_trans_array, 0, byte_size));
////
////	dim3 blocks(block_x, block_y);
////	dim3 grid(nx / block_x, ny / block_y);
////
////	printf("Launching smem kernel \n");
////	transpose_smem << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);
////	gpuErrchk(cudaDeviceSynchronize());
////
////	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
////	compare_arrays(h_ref, h_trans_array, size);
////
////	printf("Launching benchmark kernel \n");
////	cudaMemset(d_trans_array, 0, byte_size);
////	transpose_read_raw_write_column_benchmark << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);
////	gpuErrchk(cudaDeviceSynchronize());
////
////	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
////	compare_arrays(h_ref, h_trans_array, size);
////
////	printf("Launching smem padding kernel \n");
////	cudaMemset(d_trans_array, 0, byte_size);
////	transpose_smem_pad << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);
////	gpuErrchk(cudaDeviceSynchronize());
////
////	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
////	compare_arrays(h_ref, h_trans_array, size);
////
////	printf("Launching smem padding and unrolling kernel \n");
////	cudaMemset(d_trans_array, 0, byte_size);
////
////	grid.x = grid.x / 2;
////
////	transpose_smem_pad_unrolling << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);
////	gpuErrchk(cudaDeviceSynchronize());
////
////	gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
////	compare_arrays(h_ref, h_trans_array, size);
////
////	cudaFree(d_trans_array);
////	cudaFree(d_mat_array);
////	free(h_ref);
////	free(h_trans_array);
////	free(h_mat_array);
////
////	gpuErrchk(cudaDeviceReset());
////	return EXIT_SUCCESS;
////}