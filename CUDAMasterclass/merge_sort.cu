#include "merge_sort.cuh"

__global__ void mergesort_kernel(int * input, int size )
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;



}

void merge(int * arr, int l, int m, int r)
{
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	/* create temp arrays */
	int *L, *R;

	L = (int*)malloc(sizeof(int)* n1);
	R = (int*)malloc(sizeof(int)* n2);

	/* Copy data to temp arrays L[] and R[] */
	for (i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for (j = 0; j < n2; j++)
		R[j] = arr[m + 1 + j];

	/* Merge the temp arrays back into arr[l..r]*/
	i = 0; // Initial index of first subarray 
	j = 0; // Initial index of second subarray 
	k = l; // Initial index of merged subarray 
	while (i < n1 && j < n2)
	{
		if (L[i] <= R[j])
		{
			arr[k] = L[i];
			i++;
		}
		else
		{
			arr[k] = R[j];
			j++;
		}
		k++;
	}

	/* Copy the remaining elements of L[], if there are any */
	while (i < n1)
	{
		arr[k] = L[i];
		i++;
		k++;
	}

	/* Copy the remaining elements of R[], if there are any */
	while (j < n2)
	{
		arr[k] = R[j];
		j++;
		k++;
	}

	free(L);
	free(R);
}

void mergesort_cpu(int * arr, int l, int r)
{
	if (l < r)
	{
		// Same as (l+r)/2, but avoids overflow for large l and h 
		int m = l + (r - l) / 2;

		// Sort first half of the array
		mergesort_cpu(arr, l, m);
		
		// Sort second half of the array
		mergesort_cpu(arr, m + 1, r);

		//merge sorted arrays
		merge(arr, l, m, r);
	}
}

void mergesort_gpu(int * arr, int l, int r)
{
	int BYTE_SIZE = sizeof(int) * (r+1);
	
	int * gpu_input;
	cudaMalloc((void**)&gpu_input, BYTE_SIZE);
	cudaMemcpy(arr, gpu_input, BYTE_SIZE, cudaMemcpyHostToDevice);

	////mergesort_kernel << < >> > ();
	cudaDeviceSynchronize();
	cudaMemcpy(arr,gpu_input,BYTE_SIZE,cudaMemcpyDeviceToHost);
}

//// args 1 : size of the input
//int main(int argc, char** argv)
//{
//	int size = 1 << 8;
//	int SIZE_OF_ELEMENT = sizeof(int);
//	int BYTE_SIZE = size * SIZE_OF_ELEMENT;
//	clock_t cpu_start, cpu_end, gpu_start, gpu_end;
//
//	if (argc > 1)
//	{
//		size = 1 << atoi(argv[1]);
//	}
//
//	int * input, *cpu_input; 
//	input = (int*)malloc(BYTE_SIZE);
//	cpu_input = (int*)malloc(BYTE_SIZE);
//
//	if (!input)
//	{
//		printf("Error allocating memory for input array \n");
//	}
//
//	if (!cpu_input)
//	{
//		printf("Error allocating memory for cpu input array \n");
//	}
//
//	initialize(input, size, INIT_ONE_TO_TEN);
//
//	for (int i = 0; i < size; i++)
//	{
//		cpu_input[i] = input[i];
//	}
//
//	cpu_start = clock();
//	mergesort_cpu(cpu_input, 0, size -1);
//	cpu_end = clock();
//
//	gpu_start = clock();
//	mergesort_gpu(input, 0, size - 1);
//	gpu_end = clock();
//
//	compare_arrays(cpu_input, input,size);
//
//	//print_array(cpu_input, size);
//
//	free(cpu_input);
//	free(input);
//
//	cudaDeviceReset();
//
//	return 0;
//}