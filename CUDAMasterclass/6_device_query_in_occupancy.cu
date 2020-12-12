#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//int main()
//{
//	int iDev = 0;
//	cudaDeviceProp iProp;
//
//	cudaGetDeviceProperties(&iProp, iDev);
//	printf("Max threads per SM : %d \n",
//		iProp.maxThreadsPerMultiProcessor);
//
//	return 0;
//}