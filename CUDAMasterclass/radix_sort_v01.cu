#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
//#include "common.h"

void radix_sort_cpu_1(int * input, const int array_size)
{
	int * temp_a = (int*)malloc(sizeof(int)*array_size);
	int * temp_b = (int*)malloc(sizeof(int)*array_size);

	int count_a = 0;
	int count_b = 0;
	int mask = 0;

	for (int i = 0; i < 32; i++)
	{
		count_a = 0;
		count_b = 0;

		mask = 1 << i;

		for (int j = 0; j < array_size; j++)
		{
			int temp = input[j];

			if (temp& mask)
			{
				temp_b[count_b] = input[j];
				count_b++;
			}
			else
			{
				temp_a[count_a] = input[j];
				count_a++;
			}
		}

		//reorder the input depend on the result of this iteration
		for (int k = 0; k < count_a; k++)
		{
			input[k] = temp_a[k];
		}

		for (int l = 0; l < count_b; l++)
		{
			input[count_a + l] = temp_b[l];
		}

		/*for (int f = 0; f < array_size; f++)
		{
			if (!(f == (array_size - 1)))
			{
				printf("%d,", input[f]);
			}
			else
			{
				printf("%d \n", input[f]);
			}
		}*/
	}
}

__global__ void radix_sort__gpu_01(int* input, int array_size)
{
	for (int i = 0; i < 32; i++)
	{


	}
}

void run_code_radix_sort()
{
	int array_size = 1 << 24;
	int byte_array_size = sizeof(int)*array_size;

	//memory allocation and initialization of array
	int  * h_array, *h_ref;

	h_array = (int*)malloc(byte_array_size);
	h_ref = (int*)malloc(byte_array_size);

	for (int i = 0; i < array_size; i++)
	{
		h_array[i] = i % 10;
	}

	dim3 grid(32);
	dim3 block(array_size / grid.x);

	int *d_array;
	cudaMalloc((int**)&d_array, byte_array_size);

	cudaMemcpy(d_array, h_array, byte_array_size, cudaMemcpyHostToDevice);
	radix_sort__gpu_01 << <grid, block >> >(d_array, array_size);
	cudaMemcpy(h_ref, d_array, byte_array_size, cudaMemcpyDeviceToHost);

	cudaFree(d_array);
	free(h_ref);
	free(h_array);
}

//int main()
//{
//	int * int_array;
//	int array_size = 20;
//	int_array = (int*)malloc(sizeof(int)*array_size);
//	for (int i = 0; i < array_size; i++)
//	{
//		int_array[i] = i % 10;
//	}
//
//	radix_sort_cpu_1(int_array, array_size);
//	print_array_radix_sort(int_array, array_size);
//
//	free(int_array);
//	system("pause");
//	return 0;
//}