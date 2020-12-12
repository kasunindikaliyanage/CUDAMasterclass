//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <stdlib.h>
//
//__global__ void unique_index_calc_threadIdx(int * data)
//{
//	int tid = threadIdx.x;
//	printf("threadIdx.x : %d - data : %d \n", tid, data[tid]);
//}
//
//
//__global__ void unique_gid_calculation(int * data)
//{
//	int tid = threadIdx.x;
//	int offset = blockIdx.x * blockDim.x;
//	int gid = tid + offset;
//
//	printf("blockIdx.x : %d, threadIdx.x : %d - data : %d \n",
//		blockIdx.x, tid, data[gid]);
//}
//
//int main()
//{
//	int array_size = 16;
//	int array_byte_size = sizeof(int) * array_size;
//	int h_data[] = { 23,9,4,53,65,12,1,33,22,43,56,1,76,81,94,32 };
//
//	for (int i = 0; i < array_size; i++)
//	{
//		printf("%d ", h_data[i]);
//	}
//	printf("\n \n");
//
//	int * d_data;
//	cudaMalloc((void**)&d_data, array_byte_size);
//	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);
//
//	dim3 block(4);
//	dim3 grid(2);
//
//	unique_index_calc_threadIdx << < grid, block >> > (d_data);
//	cudaDeviceSynchronize();
//
//	cudaDeviceReset();
//	return 0;
//}