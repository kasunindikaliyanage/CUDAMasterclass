#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <sys/utime.h>
#include <fstream> 

#define HANDLE_NULL( a ){if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

enum INIT_PARAM{
	INIT_ZERO,INIT_RANDOM,INIT_ONE,INIT_ONE_TO_TEN,INIT_FOR_SPARSE_METRICS,INIT_0_TO_X
};

//simple initialization
void initialize(int * input, const int array_size,
	INIT_PARAM PARAM = INIT_ONE_TO_TEN, int x = 0);

void initialize(float * input, const int array_size,
	INIT_PARAM PARAM = INIT_ONE_TO_TEN);

void launch_dummmy_kernel();

//compare two arrays
void compare_arrays(int * a, int * b, int size);

//reduction in cpu
int reduction_cpu(int * input, const int size);

//compare results
void compare_results(int gpu_result, int cpu_result);

//print array
void print_array(int * input, const int array_size);

//print array
void print_array(float * input, const int array_size);

//print matrix
void print_matrix(int * matrix, int nx, int ny);

void print_matrix(float * matrix, int nx, int ny);

//get matrix
int* get_matrix(int rows, int columns);

//matrix transpose in CPU
void mat_transpose_cpu(int * mat, int * transpose, int nx, int ny);

//print_time_using_host_clock
void print_time_using_host_clock(clock_t start, clock_t end);

void printData(char *msg, int *in, const int size);

void compare_arrays(float * a, float * b, float size);

void sum_array_cpu(float* a, float* b, float *c, int size);

void print_arrays_toafile(int*, int , char* );

void print_arrays_toafile_side_by_side(float*,float*,int,char*);

void print_arrays_toafile_side_by_side(int*, int*, int, char*);

#endif // !COMMON_H