//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <time.h>
//
//void initialize_my_data(int * ip, int size)
//{
//	time_t t;
//	srand((unsigned)time(&t));
//
//	for (size_t i = 0; i < size; i++)
//	{
//		//ip[i] = (float)(rand() & 0xFF) / 10.0f;
//		ip[i] = 0;
//	}
//}
//
//__global__ void add_1_to_array(int * a, int nx)
//{
//	int ix = threadIdx.x + blockIdx.x*blockDim.x;
//	int iy = threadIdx.y + blockIdx.y*blockDim.y;
//
//	int index = iy * nx + ix;
//	a[index] += 1;
//}
//
//void print_array(int * x,int size)
//{
//	for (size_t i = 0; i < size; i++)
//	{
//		printf("%d,",x[i]);
//	}
//}
//
//void run_code()
//{
//	int element_Count = 32 * 32;
//	int nx = 32;
//	size_t number_bytes = element_Count * sizeof(float);
//
//	int *h_a, *gpu_ref;
//	h_a = (int *)malloc(number_bytes);
//	gpu_ref = (int *)malloc(number_bytes);
//
//	//initialize array with values
//	initialize_my_data(h_a, element_Count);
//	memset(gpu_ref, 0, number_bytes);
//
//	int *d_a;
//	cudaMalloc((int **)&d_a, number_bytes);
//
//	cudaMemcpy(d_a, h_a, number_bytes, cudaMemcpyHostToDevice);
//
//	dim3 block(16, 16);
//	dim3 grid(nx / block.x, (element_Count / nx) / block.y);
//
//	add_1_to_array << < grid, block >> > (d_a, nx);
//
//	//wait computation in device to finish
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(gpu_ref, d_a, number_bytes, cudaMemcpyDeviceToHost);
//	//print_array(gpu_ref,element_Count);
//
//	cudaFree(d_a);
//
//	free(h_a);
//	free(gpu_ref);
//}
//
////int main()
////{
////	run_code();
////	system("pause");
////	return 0;
////}