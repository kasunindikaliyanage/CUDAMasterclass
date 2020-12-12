#include <iostream>
#include <ctime>

using namespace std;

#include "cuda_common.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <memory>

__global__ void FindClosestGPU(float3* points, int* indices, int count)
{
	if (count <= 1) return;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < count)
	{
		float3 thisPoint = points[idx]; // every thread takes its own point
		float smallestDistSoFar = 3.40282e38f; // almost the biggest possible floating point value
		int smallestIdxSoFar = -1;

		// run through the list of all other points
		for (int i = 0; i < count; i++)
		{
			if (i == idx) continue;
			float dist = (thisPoint.x - points[i].x)*(thisPoint.x - points[i].x);
			dist += (thisPoint.y - points[i].y)*(thisPoint.y - points[i].y);
			dist += (thisPoint.z - points[i].z)*(thisPoint.z - points[i].z);
			if (dist < smallestDistSoFar)
			{
				smallestDistSoFar = dist;
				smallestIdxSoFar = i;
			}
		}
		indices[idx] = smallestIdxSoFar;
	}
}
// threads per block:
__device__ const int blockSize = 640; // faster than referring to blockDim.x

__global__ void FindClosestGPU2(float3 *points, int *indices, int count)
{
	__shared__ float3 sharedPoints[blockSize];

	if (count <= 1) return;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int indexOfClosest = -1;
	float3 thisPoint;
	float distanceToClosest = 3.40282e38f;
	if (idx < count)
		thisPoint = points[idx];
	for (int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++)
	{
		//create the shared memory block - currentBlockOfPoints 
		if (threadIdx.x + currentBlockOfPoints * blockSize < count)
			sharedPoints[threadIdx.x] = points[threadIdx.x + currentBlockOfPoints * blockSize];
		__syncthreads();

		if (idx < count)
		{
			// at each run through the shared memomry block, reset the pointer to the first element
			float *ptr = &sharedPoints[0].x;

			for (int i = 0; i < blockSize; i++)
			{
				float dist = (thisPoint.x - ptr[0])*(thisPoint.x - ptr[0])
					+ (thisPoint.y - ptr[1])*(thisPoint.y - ptr[1])
					+ (thisPoint.z - ptr[2])*(thisPoint.z - ptr[2]);
				ptr += 3;

				if ((dist < distanceToClosest) && (i + currentBlockOfPoints * blockSize < count)
					&& (i + currentBlockOfPoints * blockSize != idx))
				{
					distanceToClosest = dist;
					indexOfClosest = i + currentBlockOfPoints * blockSize;
				}
			}
		}
		__syncthreads();
	}
	if (idx < count)
	{
		indices[idx] = indexOfClosest;
	}
}


//int main()
//{
//	// number of points
//	const int count = 1000000;
//
//	// array of points
//	int* indexOfClosest = (int*)malloc(sizeof(int)* count);
//	float3 * points = (float3*)malloc(sizeof(float3)* count);
//
//	// create a list of random points
//	for (int i = 0; i < count; i++)
//	{
//		points[i].x = static_cast<float>((rand() % 10000) - 5000);
//		points[i].y = static_cast<float>((rand() % 10000) - 5000);
//		points[i].z = static_cast<float>((rand() % 10000) - 5000);
//	}
//
//	// to keep track of the shortest time so far
//	long fastest = 1000000;
//
//	float3 * d_points;
//	int * d_indices;
//
//	gpuErrchk(cudaMalloc((void**)&d_points, sizeof(float3)*count))
//	gpuErrchk(cudaMalloc((void**)&d_indices, sizeof(int)*count));
//		
//	gpuErrchk(cudaMemcpy(d_points, d_points, sizeof(float3)*count, cudaMemcpyHostToDevice));
//	gpuErrchk(cudaMemcpy(d_indices, indexOfClosest, sizeof(int)*count, cudaMemcpyHostToDevice));
//
//	dim3 blocks(1024);
//	dim3 grid(count / blocks.x + 1);
//
//	// run the algorithm 20 times
//	for (int q = 0; q < 20; q++)
//	{
//		long startTime = clock();
//
//		FindClosestGPU << < grid, blocks >> > (d_points, d_indices, count);
//		cudaDeviceSynchronize();
//
//		gpuErrchk(cudaMemcpy(indexOfClosest, d_indices, sizeof(int)*count, cudaMemcpyDeviceToHost));
//
//		long finishTime = clock();
//		std::cout << "Run " << q << " took " << (finishTime - startTime) << " ms" << endl;
//
//		// if that run was faster update fastest value
//		if ((finishTime - startTime) < fastest)
//		{
//			fastest = (finishTime - startTime);
//		}
//	}
//
//	cudaFree(d_points);
//	cudaFree(d_indices);
//
//
//	// print out the shortest run time
//	std::cout << "shortest time: " << fastest << endl;
//
//	// print the final result to screen
//	std::cout << "Final results for the first 5 points: " << endl;
//	for (int i = 0; i < 5; i++)
//	{
//		std::cout << i << " --> " << indexOfClosest[i] << endl;
//	}
//
//	free(indexOfClosest);
//	free(points);
//	cudaDeviceReset();
//	return 0;
//}