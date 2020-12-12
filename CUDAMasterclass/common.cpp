#include "common.h"

void launch_dummmy_kernel()
{

}

void print_array(int * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%d,", input[i]);
		}
		else
		{
			printf("%d \n", input[i]);
		}
	}
}

void print_array(float * input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		if (!(i == (array_size - 1)))
		{
			printf("%f,", input[i]);
		}
		else
		{
			printf("%f \n", input[i]);
		}
	}
}

void print_matrix(int * matrix, int nx, int ny)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%d ",matrix[nx * iy + ix]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_matrix(float * matrix, int nx, int ny)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%.2f ", matrix[nx * iy + ix]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_arrays_toafile_side_by_side(float*a, float*b, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " <<a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile_side_by_side(int*a, int*b, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " << a[i] << " - " << b[i] << "\n";
		}
		file.close();
	}
}

void print_arrays_toafile(int*a, int size, char* name)
{
	std::ofstream file(name);

	if (file.is_open())
	{
		for (int i = 0; i < size; i++) {
			file << i << " - " << a[i] << "\n";
		}
		file.close();
	}
}



int* get_matrix(int rows, int columns)
{
	int mat_size = rows * columns;
	int mat_byte_size = sizeof(int)*mat_size;

	int * mat = (int*)malloc(mat_byte_size);

	for (int i = 0; i < mat_size; i++)
	{
		if (i % 5 == 0)
		{
			mat[i] = i;
		}
		else
		{
			mat[i] = 0;
		}
	}

	//initialize(mat,mat_size,INIT_FOR_SPARSE_METRICS);
	return mat;
}

//simple initialization
void initialize(int * input, const int array_size,
	INIT_PARAM PARAM, int x)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		time_t t;
		srand((unsigned)time(&t));
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
				input[i] = value;
			}
			else
			{
				input[i] = 0;
			}
		}
	}
	else if (PARAM == INIT_0_TO_X)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			input[i] = (int)(rand() & 0xFF);
		}
	}
}

void initialize(float * input, const int array_size,
	INIT_PARAM PARAM)
{
	if (PARAM == INIT_ONE)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_ONE_TO_TEN)
	{
		for (int i = 0; i < array_size; i++)
		{
			input[i] = i % 10;
		}
	}
	else if (PARAM == INIT_RANDOM)
	{
		srand(time(NULL));
		for (int i = 0; i < array_size; i++)
		{
			input[i] = rand() % 10;
		}
	}
	else if (PARAM == INIT_FOR_SPARSE_METRICS)
	{
		srand(time(NULL));
		int value;
		for (int i = 0; i < array_size; i++)
		{
			value = rand() % 25;
			if (value < 5)
			{
				input[i] = value;
			}
			else
			{
				input[i] = 0;
			}
		}
	}
}

//cpu reduction
int reduction_cpu(int * input, const int size)
{
	int sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += input[i];
	}
	return sum;
}

//cpu transpose
void mat_transpose_cpu(int * mat, int * transpose, int nx, int ny)
{
	for (int  iy = 0; iy < ny; iy++)
	{
		for (int  ix = 0; ix < nx; ix++)
		{
			transpose[ix * ny + iy] = mat[iy * nx + ix];
		}
	}
}

//compare results
void compare_results(int gpu_result, int cpu_result)
{
	printf("GPU result : %d , CPU result : %d \n",
		gpu_result, cpu_result);

	if (gpu_result == cpu_result)
	{
		printf("GPU and CPU results are same \n");
		return;
	}

	printf("GPU and CPU results are different \n");
}


//compare arrays
void compare_arrays(int * a, int * b, int size)
{
	for (int  i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			printf("%d - %d | %d \n", i, a[i], b[i]);
			//return;
		}
	}
	printf("Arrays are same \n");
}

void compare_arrays(float * a, float * b, float size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			
			return;
		}
	}
	printf("Arrays are same \n");
	
}

void print_time_using_host_clock(clock_t start, clock_t end)
{
	printf("GPU kernel execution time : %4.6f \n",
		(double)((double)(end - start) / CLOCKS_PER_SEC));
}

void printData(char *msg, int *in, const int size)
{
	printf("%s: ", msg);

	for (int i = 0; i < size; i++)
	{
		printf("%5d", in[i]);
		fflush(stdout);
	}

	printf("\n");
	return;
}

void sum_array_cpu(float* a, float* b, float *c, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}
