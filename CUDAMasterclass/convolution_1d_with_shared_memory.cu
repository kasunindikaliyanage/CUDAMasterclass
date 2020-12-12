//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <iostream>
//
//#ifndef MAX_MASK_WIDTH
//#define MAX_MASK_WIDTH 10
//__constant__ float MASK[MAX_MASK_WIDTH];
//
//#define TILE_SIZE 128
//
//#endif;
//
////this kernel is example for unefficent memory access. 
////this implementation considers only 1D grid. and assumes that mask width is ODD NUMBER
////We are going to use constant memory to store the mask
//__global__ void convolution_1d_dram_and_constant_and_shared(float * input, float* output, int array_lenght, int mask_width)
//{
//	int gid = blockIdx.x*blockDim.x + threadIdx.x;
//	
//	__shared__ float s_mem[TILE_SIZE + MAX_MASK_WIDTH - 1];
//	
//	if (threadIdx.x < mask_width / 2)
//	{
//		if (gid > mask_width / 2)
//		{
//			s_mem[threadIdx.x] = input[gid - mask_width / 2];
//		}
//		else
//		{
//			s_mem[threadIdx.x] = 0.0f;
//		}
//	}
//
//	if ((threadIdx.x + mask_width/2) >= blockDim.x)
//	{
//		if (gid < (array_lenght - mask_width/2))
//		{
//			s_mem[threadIdx.x + mask_width / 2] = input[gid + mask_width/2];
//		}
//		else
//		{
//			s_mem[threadIdx.x + mask_width / 2]=0;
//		}
//	}
//
//	float temp_value = 0;
//	
//	for (int i = 0; i < mask_width; i++)
//	{
//		temp_value += s_mem[i] * MASK[i];
//	}
//
//	output[gid] = temp_value;
//}
//
//void run_code_convolution_3()
//{
//	int array_lenght = 128 * 2;
//	int mask_width = MAX_MASK_WIDTH;
//
//	int array_byte_size = sizeof(float)*array_lenght;
//	int mask_byte_size = sizeof(float)*mask_width;
//
//	float *h_input_array, *h_mask, *h_output;
//	float *d_input_array, *d_output;
//
//	//host memory allocation
//	h_input_array = (float*)malloc(array_byte_size);
//	h_output = (float*)malloc(array_byte_size);
//	h_mask = (float*)malloc(mask_byte_size);
//
//	//initialize array
//	for (int i = 0; i < array_lenght; i++)
//	{
//		h_input_array[i] = 1.0f;
//	}
//
//	//initialize mask
//	for (int i = 0; i < mask_width; i++)
//	{
//		h_mask[i] = 1.0f;
//	}
//
//	dim3 grid(array_lenght/TILE_SIZE);
//	dim3 block(TILE_SIZE);
//
//	//device memory allocation
//	cudaMalloc((float**)&d_input_array, array_byte_size);
//	cudaMalloc((float**)&d_output, array_byte_size);
//
//	//transfer the initiazed arrays to device
//	cudaMemcpy(d_input_array, h_input_array, array_byte_size, cudaMemcpyHostToDevice);
//	cudaMemcpyToSymbol(MASK, h_mask, mask_byte_size);
//
//	//kernel launch
//	convolution_1d_dram_and_constant_and_shared << <grid, block >> > (d_input_array, d_output, array_lenght, mask_width);
//	//test_kernel << <grid,block >>> (d_input_array);
//	cudaDeviceSynchronize();
//
//	//copy the output back to the host
//	cudaMemcpy(h_output, d_output, array_byte_size, cudaMemcpyDeviceToHost);
//
//	//free the device memory
//	cudaFree(d_input_array);
//	cudaFree(d_output);
//
//	//free the host memory
//	free(h_input_array);
//	free(h_output);
//	free(h_mask);
//}
////
////int main()
////{
////	run_code_convolution_3();
////	system("pause");
////	return 0;
////}