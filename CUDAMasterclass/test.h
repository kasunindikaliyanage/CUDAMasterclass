//
//
//__global__ void transpose_smem(int *in, int *out, int nx, int ny)
//{
//	// static shared memory  
//	__shared__ float tile[BDIMY][BDIMX];
//
//	int ix, iy, in_index;
//	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;
//
//	//input coordinate calculation
//	ix = blockIdx.x *blockDim.x + threadIdx.x;
//	iy = blockIdx.y *blockDim.y + threadIdx.y;
//
//	// linear global memory index for original matrix 
//	in_index = iy*nx + ix;
//
//	//1D index calculation for shared memory elements
//	_1d_index = threadIdx.y*blockDim.x + threadIdx.x;
//
//	//row and column index for shared memory column major access
//	i_row = _1d_index / blockDim.y;
//	i_col = _1d_index%blockDim.y;
//
//	// coordinate in transposed matrix
//	out_ix = blockIdx.y * blockDim.y + i_col;
//	out_iy = blockIdx.x * blockDim.x + i_row;
//
//	// linear global memory index for transposed matrix  
//	out_index = out_iy*ny + out_ix;
//
//	//printf("%d,", in[ti]);
//	// transpose with boundary test   
//	if (ix < nx && iy < ny)
//	{
//		// load data from global memory to shared memory 
//		tile[threadIdx.y][threadIdx.x] = in[in_index];
//
//		// thread synchronization     
//		__syncthreads();
//
//		// store data to global memory from shared memory  
//		out[out_index] = tile[i_col][i_row];
//	}
//}
