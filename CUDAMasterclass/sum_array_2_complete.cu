//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <time.h>
//
//void CHECK(cudaError_t error)
//{
//	if (error != cudaSuccess)
//	{
//		printf("Error : %s : %d, ", __FILE__, __LINE__);
//		printf("code : %d, reason: %s \n", error, cudaGetErrorString(error));
//		exit(1);
//	}
//}
//
//void checkResult(float *host_ref, float *gpu_ref, const int N)
//{
//	double epsilon = 0.0000001;
//	bool match = 1;
//
//	for (size_t i = 0; i < N; i++)
//	{
//		if (abs(host_ref[i] - gpu_ref[i]) > epsilon)
//		{
//			match = 0;
//			printf("Arrays do not match! \n");
//			printf("host %5.2f  gpu %5.2f at current %d\n", host_ref[i], gpu_ref[i], N);
//			break;
//		}
//	}
//
//	if (match) printf("Arrays match . \n\n");
//}
//
//void initialize_data_s(float * ip, int size)
//{
//	time_t t;
//	srand((unsigned)time(&t));
//
//	for (size_t i = 0; i < size; i++)
//	{
//		ip[i] = (float)(rand() & 0xFF) / 10.0f;
//	}
//}
//
//void sum_array_cpu(float * a, float *  b, float * c, const int N)
//{
//	for (size_t i = 0; i < N; i++)
//	{
//		c[i] = a[i] + b[i];
//	}
//}
//
//__global__ void sum_array_gpu(float * a, float *  b, float * c)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//	printf("a =%f  b = %f c = %f \n", a[i], b[i], c[i]);
//}
//
//void run_code()
//{
//	int element_Count = 32;
//	size_t number_bytes = element_Count * sizeof(float);
//
//	float *h_a, *h_b, *host_ref, *gpu_ref;
//
//	h_a = (float *)malloc(number_bytes);
//	h_b = (float *)malloc(number_bytes);
//	host_ref = (float *)malloc(number_bytes);
//	gpu_ref = (float *)malloc(number_bytes);
//
//	initialize_data_s(h_a, element_Count);
//	initialize_data_s(h_b, element_Count);
//
//	memset(host_ref, 0, number_bytes);
//	memset(gpu_ref, 0, number_bytes);
//
//	float *d_a, *d_b, *d_c;
//	cudaMalloc((float **)&d_a, number_bytes);
//	cudaMalloc((float **)&d_b, number_bytes);
//	cudaMalloc((float **)&d_c, number_bytes);
//
//	cudaMemcpy(d_a, h_a, number_bytes, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, h_b, number_bytes, cudaMemcpyHostToDevice);
//
//	dim3 block(element_Count);
//	dim3 grid(element_Count / block.x);
//
//	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c);
//
//	cudaMemcpy(gpu_ref, d_c, number_bytes, cudaMemcpyDeviceToHost);
//
//	cudaFree(d_a);
//	cudaFree(d_b);
//	cudaFree(d_c);
//
//	free(h_a);
//	free(h_b);
//	free(host_ref);
//	free(gpu_ref);
//}
//
////int main()
////{
////	run_code();
////	system("pause");
////	return 0;
////}