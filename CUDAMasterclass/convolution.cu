#include "convolution.cuh"

#define BLOCK_SIZE 128
#define MASK_SIZE 256
#define MASK_SIZE_2D 25

//constant memory declaration
__constant__ float coef[11];

//Implementation of CPU convolution 1d
//	input : 
//	mask  :
//	output:
//	input_size :
//	mask_size  :
void convolution_cpu_1d(float *input, const float *mask, float *output, int array_size, int mask_size)
{
	//Radius of the element used by one iteration from one output element
	int MASK_RADIUS = mask_size / 2;

	float temp = 0.0f;
	int ELEMENT_INDEX = 0;

	//calculate convolution value for each element in the output array
	for (int i = 0; i < array_size; i++)
	{
		//reset the accumulated value to calculate single output element
		temp = 0;

		//interate through mask
		for (int j = 0; j < mask_size; j++)
		{
			//calculation of index of the input element is going to multiply with this mask element
			ELEMENT_INDEX = i - MASK_RADIUS + j;

			//index of the input array element should be existing element.
			//index should not less than 0 or greater than input array size - 1 
			if (!(ELEMENT_INDEX < 0 || ELEMENT_INDEX >(array_size - 1)))
			{
				temp += input[ELEMENT_INDEX] * mask[j];
			}
		}

		//sets the  final accumulated value for output index
		output[i] = temp;
	}
}

void convolution_cpu_2d(float *input, const float *mask, float *output, int array_size
	, const int nx, const int ny, const int mx, const int my)
{
	int mask_size = mx * my;

	//Radius of the element used by one iteration from one output element
	int MASK_RADIUS = mx / 2;

	float temp = 0.0f;
	int ELEMENT_INDEX = 0;

	int INPUT_ROW = 0;
	int INPUT_COL = 0;
	int MASK_ROW = 0;
	int MASK_COL = 0;

	int ELEMENT_COL_INDEX = 0;
	int ELEMENT_ROW_INDEX = 0;

	//calculate convolution value for each element in the output array
	for (int i = 0; i < array_size; i++)
	{
		//reset the accumulated value to calculate single output element
		temp = 0;

		//calculating row and column of input element based on global index
		INPUT_ROW = i / nx;
		INPUT_COL = i % nx;

		//interate through mask
		for (int j = 0; j < mask_size; j++)
		{
			//calculating row and column of mask element based on linear mask index
			MASK_ROW = j / mx;
			MASK_COL = j % mx;

			ELEMENT_COL_INDEX = INPUT_COL - MASK_RADIUS + MASK_COL;
			ELEMENT_ROW_INDEX = INPUT_ROW - MASK_RADIUS + MASK_ROW;

			if (!((ELEMENT_COL_INDEX < 0 || ELEMENT_COL_INDEX >(nx - 1))
				|| (ELEMENT_ROW_INDEX < 0 || ELEMENT_ROW_INDEX >(ny - 1))))
			{
				temp += input[ELEMENT_ROW_INDEX * nx + ELEMENT_COL_INDEX] * mask[j];
			}
		}

		//sets the  final accumulated value for output index
		output[i] = temp;
	}
}

__global__ void convolution_gpu_1d_naive(float *input, float *mask, float *output, int array_size, int mask_size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int MASK_RADIUS = mask_size / 2;
	int ELEMENT_INDEX = 0;
	float temp = 0.0f;

	//for grids which have more threads than input array size
	if (gid < array_size)
	{
		for (int j = 0; j < mask_size; j++)
		{
			ELEMENT_INDEX = gid - MASK_RADIUS + j;

			//input array boundary check
			if (!(ELEMENT_INDEX < 0 || ELEMENT_INDEX >(array_size - 1)))
			{
				temp += input[ELEMENT_INDEX] * mask[j];
			}
		}
		output[gid] = temp;
	}
}

__global__ void convolution_gpu_1d_shared(float *input, float *mask, float *output, int array_size, int mask_size)
{
	__shared__ float shared_array[BLOCK_SIZE + (MASK_SIZE - 1)];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int MASK_RADIUS = mask_size / 2;

	//for grids which have more threads than input array size
	if (gid < array_size)
	{
		//load the necssasary global memory to process one thread block to shared memory
		shared_array[tid + MASK_RADIUS] = input[gid];

		//logic to assigne halo elements to first and last thread block
		if (blockIdx.x == 0 || blockIdx.x == (gridDim.x - 1))
		{
			//for first thread block
			if (blockIdx.x == 0)
			{
				if (tid < MASK_RADIUS)
				{
					shared_array[tid] = 0;
					shared_array[blockDim.x + MASK_RADIUS + tid] = input[gid + blockDim.x];
				}
			}

			//for last thread block
			if (blockIdx.x == (gridDim.x - 1))
			{
				if (tid < MASK_RADIUS)
				{
					shared_array[tid] = input[gid - MASK_RADIUS];
					shared_array[blockDim.x + MASK_RADIUS + tid] = 0;
				}
			}
		}
		else
		{
			if (tid < MASK_RADIUS)
			{
				shared_array[tid] = input[gid - MASK_RADIUS];
				shared_array[blockDim.x + MASK_RADIUS + tid] = input[gid + blockDim.x];
			}
		}

		//wait shared memory loading to finish
		__syncthreads();

		int ELEMENT_INDEX = 0;
		float temp = 0.0f;

		for (int j = 0; j < mask_size; j++)
		{
			//now the element of input arrays access from shared memory
			ELEMENT_INDEX = tid + j;
			if (ELEMENT_INDEX < (blockDim.x + MASK_SIZE))
				temp += shared_array[ELEMENT_INDEX] * mask[j];
		}
		output[gid] = temp;
	}
}

__global__ void convolution_gpu_1d_shared_constant(float *input, float *output, int array_size, int mask_size)
{
	__shared__ float shared_array[BLOCK_SIZE + (MASK_SIZE - 1)];

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	const int MASK_RADIUS = mask_size / 2;

	//for grids which have more threads than input array size
	if (gid < array_size)
	{
		//load the necssasary global memory to process one thread block to shared memory
		shared_array[tid + MASK_RADIUS] = input[gid];

		//logic to assigne halo elements to first and last thread block
		if (blockIdx.x == 0 || blockIdx.x == (gridDim.x - 1))
		{
			//for first thread block
			if (blockIdx.x == 0)
			{
				if (tid < MASK_RADIUS)
				{
					shared_array[tid] = 0;
					shared_array[blockDim.x + MASK_RADIUS + tid] = input[gid + blockDim.x];
				}
			}

			//for last thread block
			if (blockIdx.x == (gridDim.x - 1))
			{
				if (tid < MASK_RADIUS)
				{
					shared_array[tid] = input[gid - MASK_RADIUS];
					shared_array[blockDim.x + MASK_RADIUS + tid] = 0;
				}
			}
		}
		else
		{
			if (tid < MASK_RADIUS)
			{
				shared_array[tid] = input[gid - MASK_RADIUS];
				shared_array[blockDim.x + MASK_RADIUS + tid] = input[gid + blockDim.x];
			}
		}

		//wait shared memory loading to finish
		__syncthreads();

		int ELEMENT_INDEX = 0;
		float temp = 0.0f;

		for (int j = 0; j < mask_size; j++)
		{
			//now the element of input arrays access from shared memory
			ELEMENT_INDEX = tid + j;
			if (ELEMENT_INDEX < (blockDim.x + MASK_SIZE))
				temp += shared_array[ELEMENT_INDEX] * coef[j];
		}
		output[gid] = temp;
	}
}

__global__ void convolution_gpu_2d_naive(float *input, const float *mask, float *output, int array_size
	, int mask_size, int nx, int ny, int mx, int my)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	//Radius of the element used by one iteration from one output element
	int MASK_RADIUS = mx / 2;

	float temp = 0.0f;
	int ELEMENT_INDEX = 0;

	int INPUT_ROW = 0;
	int INPUT_COL = 0;
	int MASK_ROW = 0;
	int MASK_COL = 0;

	int ELEMENT_COL_INDEX = 0;
	int ELEMENT_ROW_INDEX = 0;

	if (gid < array_size)
	{
		//calculating row and column of input element based on global index
		INPUT_ROW = gid / nx;
		INPUT_COL = gid % nx;

		//interate through mask
		for (int j = 0; j < mask_size; j++)
		{
			//calculating row and column of mask element based on linear mask index
			MASK_ROW = j / mx;
			MASK_COL = j % mx;

			ELEMENT_COL_INDEX = INPUT_COL - MASK_RADIUS + MASK_COL;
			ELEMENT_ROW_INDEX = INPUT_ROW - MASK_RADIUS + MASK_ROW;

			if (!((ELEMENT_COL_INDEX < 0 || ELEMENT_COL_INDEX >(nx - 1))
				|| (ELEMENT_ROW_INDEX < 0 || ELEMENT_ROW_INDEX >(ny - 1))))
			{
				temp += input[ELEMENT_ROW_INDEX * nx + ELEMENT_COL_INDEX] * mask[j];
			}
		}

		//sets the  final accumulated value for output index
		output[gid] = temp;
	}
}

__global__ void convolution_gpu_2d_shared(float *input, const float *mask, float *output, int array_size
	, int mask_size, int nx, int ny, int mx, int my)
{
	extern __shared__ float tile[];

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	//Radius of the element used by one iteration from one output element
	int MASK_RADIUS = mx / 2;	

	//row size of the shared memory
	int TILE_ROW_SIZE = BLOCK_SIZE + MASK_RADIUS * 2;

	float temp = 0.0f;
	int ELEMENT_INDEX = 0;

	int INPUT_ROW = 0;
	int INPUT_COL = 0;
	int MASK_ROW = 0;
	int MASK_COL = 0;

	int INPUT_INDEX = 0;
	int TILE_INDEX = 0;

	int SMEM_ROW_INDEX = 0;
	int SMEM_COL_INDEX = 0;

	int ELEMENT_COL_INDEX = 0;
	int ELEMENT_ROW_INDEX = 0;


	if (gid < array_size)
	{
		////directly assign this data block to shared memory
		//tile[tid + MASK_RADIUS + MASK_RADIUS * TILE_ROW_SIZE] = input[gid];

		INPUT_ROW = gid / nx;
		INPUT_COL = gid % nx;

		for (int i = 0; i < my; i++)
		{
			SMEM_ROW_INDEX = INPUT_ROW - MASK_RADIUS + i;
			SMEM_COL_INDEX = INPUT_COL - MASK_RADIUS;

			if (!(SMEM_ROW_INDEX < 0 ||  SMEM_ROW_INDEX > (ny-1)) )
			{
				tile[tid + MASK_RADIUS + i * TILE_ROW_SIZE] = input[gid - (MASK_RADIUS - i) * BLOCK_SIZE ];

				if (tid < MASK_RADIUS)
				{
					if (!(SMEM_COL_INDEX < 0))
					{
						tile[tid + i * TILE_ROW_SIZE] = input[gid - (MASK_RADIUS - tid) - (MASK_RADIUS - i) * BLOCK_SIZE];
					}
					else
					{
						tile[tid + i * TILE_ROW_SIZE] = 0;
					}

					SMEM_COL_INDEX = INPUT_COL + BLOCK_SIZE;

					if (!(SMEM_COL_INDEX > (nx-1)))
					{
						tile[tid + MASK_RADIUS + BLOCK_SIZE + i * TILE_ROW_SIZE] = input[gid + BLOCK_SIZE - (MASK_RADIUS - i) * BLOCK_SIZE];
					}
					else
					{
						tile[tid + MASK_RADIUS + BLOCK_SIZE + i * TILE_ROW_SIZE] = 0;
					}
				}
			}
			else
			{
				tile[tid + MASK_RADIUS + i * TILE_ROW_SIZE] = 0;

				if (tid < MASK_RADIUS)
				{
					tile[tid + i * TILE_ROW_SIZE] = 0;
					tile[tid + MASK_RADIUS + BLOCK_SIZE + i * TILE_ROW_SIZE] = 0;
				}
			}
		}

		//wait until shared memory load is finished
		__syncthreads();

		for (int j = 0; j < mask_size; j++)
		{
			//calculating row and column of mask element based on linear mask index
			MASK_ROW = j / mx;
			MASK_COL = j % mx;

			ELEMENT_COL_INDEX = tid + MASK_COL;
			ELEMENT_ROW_INDEX = MASK_ROW;

			temp += tile[ELEMENT_ROW_INDEX * TILE_ROW_SIZE + ELEMENT_COL_INDEX] * mask[j];
		}

		//sets the  final accumulated value for output index
		output[gid] = temp;
	}
}

__global__ void convolution_gpu_2d_shared_intrinsic(float *input, const float *mask, float *output, int array_size
	, int mask_size, int nx, int ny, int mx, int my)
{
	extern __shared__ float tile[];

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	//Radius of the element used by one iteration from one output element
	int MASK_RADIUS = mx / 2;

	//row size of the shared memory
	int TILE_ROW_SIZE = BLOCK_SIZE + MASK_RADIUS * 2;

	float temp = 0.0f;
	int ELEMENT_INDEX = 0;

	int INPUT_ROW = 0;
	int INPUT_COL = 0;
	int MASK_ROW = 0;
	int MASK_COL = 0;

	int INPUT_INDEX = 0;
	int TILE_INDEX = 0;

	int SMEM_ROW_INDEX = 0;
	int SMEM_COL_INDEX = 0;

	int ELEMENT_COL_INDEX = 0;
	int ELEMENT_ROW_INDEX = 0;


	if (gid < array_size)
	{
		////directly assign this data block to shared memory
		//tile[tid + MASK_RADIUS + MASK_RADIUS * TILE_ROW_SIZE] = input[gid];

		INPUT_ROW = gid / nx;
		INPUT_COL = gid % nx;

		for (int i = 0; i < my; i++)
		{
			SMEM_ROW_INDEX = INPUT_ROW - MASK_RADIUS + i;
			SMEM_COL_INDEX = INPUT_COL - MASK_RADIUS;

			if (!(SMEM_ROW_INDEX < 0 || SMEM_ROW_INDEX >(ny - 1)))
			{
				tile[tid + MASK_RADIUS + i * TILE_ROW_SIZE] = input[gid - (MASK_RADIUS - i) * BLOCK_SIZE];

				if (tid < MASK_RADIUS)
				{
					if (!(SMEM_COL_INDEX < 0))
					{
						tile[tid + i * TILE_ROW_SIZE] = input[gid - (MASK_RADIUS - tid) - (MASK_RADIUS - i) * BLOCK_SIZE];
					}
					else
					{
						tile[tid + i * TILE_ROW_SIZE] = 0;
					}

					SMEM_COL_INDEX = INPUT_COL + BLOCK_SIZE;

					if (!(SMEM_COL_INDEX > (nx - 1)))
					{
						tile[tid + MASK_RADIUS + BLOCK_SIZE + i * TILE_ROW_SIZE] = input[gid + BLOCK_SIZE - (MASK_RADIUS - i) * BLOCK_SIZE];
					}
					else
					{
						tile[tid + MASK_RADIUS + BLOCK_SIZE + i * TILE_ROW_SIZE] = 0;
					}
				}
			}
			else
			{
				tile[tid + MASK_RADIUS + i * TILE_ROW_SIZE] = 0;

				if (tid < MASK_RADIUS)
				{
					tile[tid + i * TILE_ROW_SIZE] = 0;
					tile[tid + MASK_RADIUS + BLOCK_SIZE + i * TILE_ROW_SIZE] = 0;
				}
			}
		}

		//wait until shared memory load is finished
		__syncthreads();

		for (int j = 0; j < mask_size; j++)
		{
			//calculating row and column of mask element based on linear mask index
			MASK_ROW = j / mx;
			MASK_COL = j % mx;

			ELEMENT_COL_INDEX = tid + MASK_COL;
			ELEMENT_ROW_INDEX = MASK_ROW;

			temp += tile[ELEMENT_ROW_INDEX * TILE_ROW_SIZE + ELEMENT_COL_INDEX] * mask[j];
		}

		//sets the  final accumulated value for output index
		output[gid] = temp;
	}
}

//kernel 0 - naive // 1 - shared
void convolution_gpu_1d_launch(float *input, const float *mask, float *output, int array_size, int mask_size, int kernel)
{
	//number of bytes needed to hold each array
	int byte_size = sizeof(float)* array_size;
	int mask_byte_size = sizeof(float) * mask_size;

	//device pointers declaration
	float *d_input, *d_mask, *d_output;

	//devie pointers memory allocation
	cudaMalloc((void**)&d_input, byte_size);
	cudaMalloc((void**)&d_output, byte_size);
	cudaMalloc((void**)&d_mask, mask_byte_size);

	//copying the host arrays to device
	cudaMemcpy(d_input, input, byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, mask_byte_size, cudaMemcpyHostToDevice);

	//kernel launch
	dim3 block(BLOCK_SIZE);
	dim3 grid((array_size / block.x));

	if (kernel == 0)
	{
		convolution_gpu_1d_naive << <grid, block >> > (d_input, d_mask, d_output, array_size, mask_size);
	}
	else if (kernel == 1)
	{
		convolution_gpu_1d_shared << <grid, block >> > (d_input, d_mask, d_output, array_size, mask_size);
	}
	cudaDeviceSynchronize();

	//copy the memory back to host
	cudaMemcpy(output, d_output, byte_size, cudaMemcpyDeviceToHost);

	cudaFree(d_output);
	cudaFree(d_mask);
	cudaFree(d_output);
}

void convolution_gpu_1d_constant_lauch(float *input, const float *mask, float *output, int array_size, int mask_size)
{
	//number of bytes needed to hold each array
	int byte_size = sizeof(float)* array_size;

	//device pointers declaration
	float *d_input, *d_output;

	//devie pointers memory allocation
	cudaMalloc((void**)&d_input, byte_size);
	cudaMalloc((void**)&d_output, byte_size);

	cudaError error = cudaMemcpyToSymbol(coef, mask, (11) * sizeof(float));

	//copying the host arrays to device
	cudaMemcpy(d_input, input, byte_size, cudaMemcpyHostToDevice);

	//kernel launch
	dim3 block(BLOCK_SIZE);
	dim3 grid((array_size / block.x));

	convolution_gpu_1d_shared_constant << <grid, block >> > (d_input, d_output, array_size, mask_size);
	cudaDeviceSynchronize();

	//copy the memory back to host
	cudaMemcpy(output, d_output, byte_size, cudaMemcpyDeviceToHost);

	cudaFree(d_output);
	cudaFree(d_output);

}

void convolution_gpu_2d_launch(float *input, const float *mask, float *output, int array_size, int mask_size,
	const int nx, const int ny, const int mx, const int my)
{
	//number of bytes needed to hold each array
	int byte_size = sizeof(float)* array_size;
	int mask_byte_size = sizeof(float) * mask_size;

	//device pointers declaration
	float *d_input, *d_mask, *d_output;

	//devie pointers memory allocation
	cudaMalloc((void**)&d_input, byte_size);
	cudaMalloc((void**)&d_output, byte_size);
	cudaMalloc((void**)&d_mask, mask_byte_size);

	//copying the host arrays to device
	cudaMemcpy(d_input, input, byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, mask_byte_size, cudaMemcpyHostToDevice);

	//kernel launch
	dim3 block(BLOCK_SIZE);
	dim3 grid((array_size / block.x)+ 1);
	
	convolution_gpu_2d_shared<< <grid, block, sizeof(float)* ((BLOCK_SIZE + 2 * (mx - 1)) * my) >> > (d_input, d_mask, 
		d_output, array_size, mask_size, nx, ny, mx, my);

	cudaDeviceSynchronize();

	//copy the memory back to host
	cudaMemcpy(output, d_output, byte_size, cudaMemcpyDeviceToHost);

	cudaFree(d_output);
	cudaFree(d_mask);
	cudaFree(d_output);
}

void convolution_1d(int argc, char** argv)
{
	int input_size = 1 << 25;
	const int mask_size = MASK_SIZE;
	int kernel = 1;

	if (argc > 1)
		input_size = 1 << atoi(argv[1]);

	if (argc > 2)
		kernel = atoi(argv[2]);


	//calculate the needed byte size to hold input array
	int byte_size = sizeof(float)* input_size;

	float * input_array = (float*)malloc(byte_size);
	float * cpu_output_array = (float*)malloc(byte_size);
	float * gpu_output_array = (float*)malloc(byte_size);

	const float mask[mask_size] = { 1.0,2.0,3.0,4.0,5.0,6.0,5.0,4.0,3.0,2.0,1.0 };
	//float mask[mask_size] = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };

	initialize(input_array, input_size, INIT_RANDOM);

	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	cpu_start = clock();
	convolution_cpu_1d(input_array, mask, cpu_output_array, input_size, mask_size);
	cpu_end = clock();

	gpu_start = clock();
	//calculating convolution GPU using nieve kernel
	if (kernel == 0)
	{
		//printf("Launching naive GPU implementation \n");
		//gpu_start = clock();
		convolution_gpu_1d_launch(input_array, mask, gpu_output_array, input_size, mask_size, 0);
	}
	else if (kernel == 1)
	{
		//printf("Launching shared GPU implementation \n");
		//gpu_start = clock();
		convolution_gpu_1d_launch(input_array, mask, gpu_output_array, input_size, mask_size, 1);
	}
	else if (kernel == 2)
	{
		//printf("Launching shared constant GPU implementation \n");
		//gpu_start = clock();
		convolution_gpu_1d_constant_lauch(input_array, mask, gpu_output_array, input_size, mask_size);
	}
	gpu_end = clock();

	compare_arrays(gpu_output_array, cpu_output_array, input_size);

	//time calculation
	double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
	double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC;

	printf("Convolution CPU execution time : %f \n", cpu_time);
	printf("Convolution GPU execution time : %f \n", gpu_time);

	free(cpu_output_array);
	free(gpu_output_array);
	free(input_array);

	cudaDeviceReset();
}

void convolution_2d(int argc, char** argv)
{
	int input_size = 1 << 22;
	const int mask_size = MASK_SIZE_2D;

	const int nx = 1 << 12;
	const int ny = 1 << 10;
	const int mx = 5;
	const int my = 5;

	int byte_size = sizeof(float)* input_size;

	float * input_array = (float*)malloc(byte_size);
	float * cpu_output_array = (float*)malloc(byte_size);
	float * gpu_output_array = (float*)malloc(byte_size);

	float * mask = (float*)malloc(MASK_SIZE_2D * sizeof(float));

	initialize(mask,mask_size ,INIT_ONE);

	initialize(input_array, input_size, INIT_ONE);

	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	cpu_start = clock();
	convolution_cpu_2d(input_array, mask, cpu_output_array, input_size, nx, ny, mx, my);
	cpu_end = clock();

	printf("CPU computation is finished \n");

	gpu_start = clock();
	convolution_gpu_2d_launch(input_array, mask, gpu_output_array, input_size, mask_size, nx, ny, mx, my);
	gpu_end = clock();

	compare_arrays(gpu_output_array,cpu_output_array,input_size);

	//time calculation
	double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
	double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC;

	printf("Convolution CPU execution time : %f \n", cpu_time);
	printf("Convolution GPU execution time : %f \n", gpu_time);
	printf("Speed up : %f \n",cpu_time / gpu_time);

	//print_matrix(cpu_output_array, nx, ny);
	
	cudaDeviceReset();
}

//int main(int argc, char** argv)
//{
//	convolution_2d(argc, argv);
//}