//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <iostream>
//
////this kernel is example for unefficent memory access. 
////this implementation considers only 1D grid. and assumes that mask width is odd number
//__global__ void convolution_1d_dram(float * input, float* mask, float* output, int array_lenght, int mask_width)
//{
//	int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
//	float temp_value = 0;
//
//	////we can assigne thread_index-mask_width/2 to automatic variable to save memory bandwidth as shown in following code segment
//	int offset = thread_index - mask_width / 2;
//	for (int i = 0; i < mask_width; i++)
//	{
//		if ((offset + i) >= 0 && (offset + i) < array_lenght)
//		{
//			temp_value += input[offset + i] * mask[i];
//		}
//	}
//
//	/*for (int i = 0; i < mask_width; i++)
//	{
//		if ((thread_index+i-mask_width/2)>= 0 && (thread_index + i- mask_width / 2) < array_lenght)
//		{
//			temp_value += input[thread_index + i - mask_width / 2] * mask[i];
//		}
//	}
//	*/
//	output[thread_index] = temp_value;
//	//printf("%d thread value %f \n", thread_index, temp_value);
//}
//
//void print_array(float* my_array, int size)
//{
//	for (int i = 0; i < size; i++)
//	{
//		if (i % 32 == 0)
//		{
//			printf("\n");
//		}
//		printf("%.2f,",my_array[i]);
//	}
//
//	printf("\n");
//}
//
//void run_code_convolution_1()
//{
//	int array_lenght = 128 * 2;
//	int mask_width = 32;
//
//	int array_byte_size = sizeof(float)*array_lenght;
//	int mask_byte_size = sizeof(float)*mask_width;
//
//	float *h_input_array, *h_mask, *h_output;
//	float *d_input_array, *d_mask, *d_output;
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
//	dim3 grid(32);
//	dim3 block((array_lenght) / grid.x);
//
//	//device memory allocation
//	cudaMalloc((float**)&d_input_array, array_byte_size);
//	cudaMalloc((float**)&d_mask, mask_byte_size);
//	cudaMalloc((float**)&d_output, array_byte_size);
//
//	//transfer the initiazed arrays to device
//	cudaMemcpy(d_input_array, h_input_array, array_byte_size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_mask, h_mask, mask_byte_size, cudaMemcpyHostToDevice);
//
//	//kernel launch
//	convolution_1d_dram << <grid, block >> > (d_input_array, d_mask, d_output, array_lenght, mask_width);
//	//test_kernel << <grid,block >>> (d_input_array);
//	cudaDeviceSynchronize();
//
//	//copy the output back to the host
//	cudaMemcpy(h_output, d_output, array_byte_size, cudaMemcpyDeviceToHost);
//
//	//print the arrays
//	//printf("Printing input :\n ");
//	//print_array(h_input_array,array_lenght);
//
//	//printf("Printing mask :\n ");
//	//print_array(h_mask, mask_width);
//
//	/*printf("Printing output :\n ");
//	print_array(h_output, array_lenght);*/
//
//	//free the device memory
//	cudaFree(d_input_array);
//	cudaFree(d_output);
//	cudaFree(d_mask);
//
//	//free the host memory
//	free(h_input_array);
//	free(h_output);
//	free(h_mask);
//}
//
////int main()
////{
////	run_code_convolution_1();
////	system("pause");
////	return 0;
////}