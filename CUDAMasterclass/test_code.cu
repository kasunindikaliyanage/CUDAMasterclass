//__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata,
//	unsigned int isize)
//{
//	int tid = threadIdx.x;
//
//	int *idata = g_idata + blockIdx.x*blockDim.x;
//	int *odata = &g_odata[blockIdx.x];
//
//	// stop condition   
//	if (isize == 2 && tid == 0)
//	{
//		g_odata[blockIdx.x] = idata[0] + idata[1];
//		return;
//	}
//
//	// nested invocation   
//	int istride = isize >> 1;
//	if (istride > 1 && tid < istride)
//	{
//		// in place reduction    
//		idata[tid] += idata[tid + istride];
//	}
//
//	// sync at block level   
//	__syncthreads();
//
//	// nested invocation to generate child grids 
//	if (tid == 0)
//	{
//		gpuRecursiveReduce << <1, istride >> > (idata, odata, istride);
//		cudaDeviceSynchronize();
//	}
//
//	// sync at block level again 
//	__syncthreads();
//}
//
//__global__ void gpuRecursiveReduce_improved(int *g_idata, int *g_odata,
//	unsigned int isize)
//{
//	int tid = threadIdx.x;
//
//	int *idata = g_idata + blockIdx.x*blockDim.x;
//	int *odata = &g_odata[blockIdx.x];
//
//	// stop condition   
//	if (isize == 2 && tid == 0)
//	{
//		g_odata[blockIdx.x] = idata[0] + idata[1];
//		return;
//	}
//
//	// nested invocation   
//	int istride = isize >> 1;
//	if (istride > 1 && tid < istride)
//	{
//		// in place reduction    
//		idata[tid] += idata[tid + istride];
//	}
//
//	// nested invocation to generate child grids 
//	if (tid == 0)
//	{
//		gpuRecursiveReduce << <1, istride >> > (idata, odata, istride);
//	}
//}
//
//__global__ void gpuRecursiveReduce_2(int *g_idata, int *g_odata, int iStride,
//	int const iDim)
//{
//	// convert global data pointer to the local pointer of this block
//	int *idata = g_idata + blockIdx.x * iDim;
//
//	// stop condition
//	if (iStride == 1 && threadIdx.x == 0)
//	{
//		g_odata[blockIdx.x] = idata[0] + idata[1];
//		return;
//	}
//
//	// in place reduction
//	idata[threadIdx.x] += idata[threadIdx.x + iStride];
//
//	// nested invocation to generate child grids
//	if (threadIdx.x == 0 && blockIdx.x == 0)
//	{
//		gpuRecursiveReduce_2 << <gridDim.x, iStride / 2 >> >(g_idata, g_odata,
//			iStride / 2, iDim);
//	}
//}