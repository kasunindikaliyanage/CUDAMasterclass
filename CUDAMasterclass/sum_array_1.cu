//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <time.h>
//
//void initialize_data(float * ip, int size)
//{
//	time_t t;
//	srand((unsigned) time(&t));
//
//	for (size_t i = 0; i < size; i++)
//	{
//		ip[i] = (float)(rand() & 0xFF) / 10.0f;
//	}
//}
//
//__global__ void sum_array(float * a, float *  b, float * c)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//	printf("a =%f  b = %f c = %f \n",a[i],b[i],c[i]);
//}
//
////int main()
////{
////	int element_Count = 32;
////
////	size_t number_bytes = element_Count * sizeof(float);
////
////	float *h_a, *h_b, *host_ref, *gpu_ref;
////
////	h_a = (float *)malloc(number_bytes);
////	h_b = (float *)malloc(number_bytes);
////	host_ref =  (float *)malloc(number_bytes);
////	gpu_ref = (float *)malloc(number_bytes);
////
////	initialize_data(h_a,element_Count);
////	initialize_data(h_b, element_Count);
////
////	memset(host_ref,0,number_bytes);
////	memset(gpu_ref,0,number_bytes);
////
////	float *d_a, *d_b, *d_c;
////	cudaMalloc((float **)&d_a,number_bytes);
////	cudaMalloc((float **)&d_b, number_bytes);
////	cudaMalloc((float **)&d_c, number_bytes);
////
////	cudaMemcpy(d_a,h_a,number_bytes,cudaMemcpyHostToDevice);
////	cudaMemcpy(d_b, h_b ,number_bytes, cudaMemcpyHostToDevice);
////
////	dim3 block(element_Count);
////	dim3 grid(element_Count/block.x);
////
////	sum_array << <grid,block >> > (d_a,d_b,d_c);
////
////	cudaMemcpy(gpu_ref,d_c,number_bytes,cudaMemcpyDeviceToHost);
////
////	cudaFree(d_a);
////	cudaFree(d_b);
////	cudaFree(d_c);
////
////	free(h_a);
////	free(h_b);
////	free(host_ref);
////	free(gpu_ref);
////
////	system("pause");
////	return 0;
////}