//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <cuda.h>
//#include <stdio.h>
//#include <iostream>
//#include <time.h>
//#include "common.h"
////#include "cuda_common.cuh"
//
//#define HISTOGRAM_BUCKET_SIZE 10
//
////---------------------------------------------------KERNELS AND CPU IMPLEMENTATIONS-----------------------------//
//
////in this kernel we expect grid with only single block
//__global__ void histogram_gpu_v01(int* input, int*output, int input_size)
//{
//	atomicAdd(&(output[input[threadIdx.x]]),1);
//}
//
//// in this kernel we expect 1D grid with multiple 1D blocks 
//// reduce the global memory writes by introducing shared memory intermideate store
//__global__ void histogram_gpu_v02(int* input, int*output, int input_size)
//{
//	__shared__  int block_output[HISTOGRAM_BUCKET_SIZE];
//	
//	int gid = blockIdx.x*blockDim.x + threadIdx.x;
//
//	atomicAdd(&(block_output[input[gid]]), 1);
//	__syncthreads();
//
//	if (threadIdx.x < HISTOGRAM_BUCKET_SIZE)
//	{
//		atomicAdd(&(output[threadIdx.x]), block_output[threadIdx.x]);
//	}
//}
//
////crating thread for to represent each element in the array may be inefficient
////so change the kernel so that single threads handles multiple elements 
//__global__ void histogram_gpu_v03(int* input, int*output, int input_size)
//{
//	//to be implements
//}
//
//void histogram_cpu(int * input, int* output, int input_size)
//{
//	for (int i = 0; i < input_size; i++)
//	{
//		output[input[i]]++;
//	}
//}
//
//
////---------------------------------------------------RUNING FUNCTIONS--------------------------------------------//
//
//void run_histogram_cpu(int input_size, int histogram_buckets)
//{
//	int * input, *output;
//	int input_byte_size = sizeof(int)*input_size;
//	int histogram_bucket_byte_size = sizeof(int)*histogram_buckets;
//
//	input = (int*)malloc(input_byte_size);
//	output = (int*)malloc(histogram_bucket_byte_size);
//	memset(output, 0, histogram_bucket_byte_size);
//
//	initialize(input, input_size);
//	printf("Printing input array \n");
//	print_array(input,input_size);
//
//	histogram_cpu(input,output,input_size);
//	printf("Printing histogram array \n");
//	print_array(output, histogram_buckets);
//	
//	free(output);
//	free(input);
//}
//
//void run_histogram_gpu(int input_size, int histogram_buckets)
//{
//	int * h_input, *h_ref;
//	int input_byte_size = sizeof(int)*input_size;
//	int histogram_bucket_byte_size = sizeof(int)*histogram_buckets;
//
//	h_input = (int*)malloc(input_byte_size);
//	h_ref = (int*)malloc(histogram_bucket_byte_size);;
//	
//	initialize(h_input, input_size);
//
//	int * d_input, *d_output;
//	cudaMalloc((int**)&d_input, input_byte_size);
//	cudaMalloc((int**)&d_output,histogram_bucket_byte_size);
//
//	dim3 grid(4);
//	dim3 block(input_size/grid.x);
//
//	cudaMemset(d_output, 0, histogram_bucket_byte_size);
//	cudaMemcpy(d_input, h_input, input_byte_size, cudaMemcpyHostToDevice);
//
//	histogram_gpu_v02 << <grid,block >> > (d_input, d_output, input_size);
//	cudaDeviceSynchronize();
//
//	cudaMemcpy(h_ref, d_output, histogram_bucket_byte_size, cudaMemcpyDeviceToHost);
//
//	print_array(h_ref,histogram_buckets);
//
//	cudaFree(d_output);
//	cudaFree(d_input);
//
//	free(h_ref);
//	free(h_input);
//}
//
////int main()
////{
////	printf("--------------------RUNNING HISTOGRAM EXAMPLE------------------------- \n");
////	int input_size = 1024; 
////	int histogram_buckets = 10;
////	run_histogram_gpu(input_size,histogram_buckets);
////
////	system("pause");
////	return 0;
////}