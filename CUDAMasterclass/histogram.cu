#include "histogram.cuh"

#define SIZE    (100*1024*1024)
#define HISTO_SIZE 256

__global__ void histogram_shared_mem(int *input,  int *histo, long size)
{
	__shared__  int temp[256];
	
	temp[threadIdx.x] = 0;
	__syncthreads();

	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int temp_id = gid;
	
	while (temp_id < size)
	{
		atomicAdd(&temp[input[temp_id]], 1);
		temp_id += stride;
	}

	__syncthreads();
	atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

__global__ void histogram_basic(int *input, int *histo, long size)
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	int temp_id = gid;

	while (temp_id < size)
	{
		atomicAdd(&histo[input[temp_id]], 1);
		temp_id += stride;
	}
}

void histogram_cpu(int* input, int* h_histo_cpu, int size)
{
	// capture the start time
	clock_t  start, stop;
	start = clock();

	for (int i = 0; i < 256; i++)
		h_histo_cpu[i] = 0;

	for (int i = 0; i < SIZE; i++)
		h_histo_cpu[input[i]]++;

	stop = clock();
	float elapsedTime = (float)(stop - start) /(float)CLOCKS_PER_SEC * 1000.0f;
	printf("Histogram CPU execution time:  %3.1f ms\n", elapsedTime);
}

void histogram_gpu(int* h_input, int* h_histo_gpu, int size)
{
	long byte_size = SIZE * sizeof(int);
	int histo_byte_size = HISTO_SIZE * sizeof(int);

	cudaEvent_t start, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	gpuErrchk(cudaEventRecord(start, 0));

	// allocate memory on the GPU for the file's data
	int * d_input, *d_histo;

	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&d_histo, histo_byte_size));
	gpuErrchk(cudaMemset(d_histo, 0, histo_byte_size));

	// kernel launch - 2x the number of mps gave best timing
	cudaDeviceProp  prop;
	gpuErrchk(cudaGetDeviceProperties(&prop, 0));
	int blocks = prop.multiProcessorCount;

	dim3 block(HISTO_SIZE);
	//dim3 grid(blocks * 32);
	dim3 grid(SIZE / block.x +1);

	histogram_basic << <grid, block >> > (d_input, d_histo, SIZE);

	gpuErrchk(cudaMemcpy(h_histo_gpu, d_histo, histo_byte_size, cudaMemcpyDeviceToHost));

	// get stop time, and display the timing results
	gpuErrchk(cudaEventRecord(stop, 0));
	gpuErrchk(cudaEventSynchronize(stop));

	float  elapsedTime;
	gpuErrchk(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Histogram GPU execution time :  %3.1f ms\n", elapsedTime);

	gpuErrchk(cudaEventDestroy(start));
	gpuErrchk(cudaEventDestroy(stop));

	cudaFree(d_histo);
	cudaFree(d_input);

	gpuErrchk(cudaDeviceReset());
}

void histogram_gpu_multistreams(int* h_input, int* h_histo_gpu, int size)
{
	long byte_size = SIZE * sizeof(int);
	int histo_byte_size = HISTO_SIZE * sizeof(int);

	cudaEvent_t start, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	gpuErrchk(cudaEventRecord(start, 0));

	// allocate memory on the GPU for the file's data
	int * d_input, *d_histo;

	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&d_histo, histo_byte_size));
	gpuErrchk(cudaMemset(d_histo, 0, histo_byte_size));

	// kernel launch - 2x the number of mps gave best timing
	cudaDeviceProp  prop;
	gpuErrchk(cudaGetDeviceProperties(&prop, 0));
	int blocks = prop.multiProcessorCount;

	dim3 block(HISTO_SIZE);
	//dim3 grid(blocks * 32);
	dim3 grid(SIZE / block.x + 1);

	histogram_basic << <grid, block >> > (d_input, d_histo, SIZE);

	gpuErrchk(cudaMemcpy(h_histo_gpu, d_histo, histo_byte_size, cudaMemcpyDeviceToHost));

	// get stop time, and display the timing results
	gpuErrchk(cudaEventRecord(stop, 0));
	gpuErrchk(cudaEventSynchronize(stop));

	float  elapsedTime;
	gpuErrchk(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Histogram GPU execution time :  %3.1f ms\n", elapsedTime);

	gpuErrchk(cudaEventDestroy(start));
	gpuErrchk(cudaEventDestroy(stop));

	cudaFree(d_histo);
	cudaFree(d_input);

	gpuErrchk(cudaDeviceReset());
}


//int main(void)
//{
//	long byte_size = SIZE * sizeof(int);
//	int histo_byte_size = HISTO_SIZE * sizeof(int);
//
//	int *h_input, *h_histo_cpu, *h_histo_gpu;
//	
//	h_input = (int*)malloc(byte_size);
//	h_histo_cpu = (int*)malloc(histo_byte_size);
//	h_histo_gpu = (int*)malloc(histo_byte_size);
//
//	//initialize the array from 0 to 255
//	initialize(h_input, SIZE, INIT_0_TO_X, 256);
//
//	histogram_cpu(h_input, h_histo_cpu, SIZE);
//	histogram_gpu(h_input, h_histo_gpu, SIZE);
//
//	compare_arrays(h_histo_gpu, h_histo_cpu, HISTO_SIZE);
//	
//	free(h_histo_gpu);
//	free(h_histo_cpu);
//	free(h_input);
//	return 0;
//}
