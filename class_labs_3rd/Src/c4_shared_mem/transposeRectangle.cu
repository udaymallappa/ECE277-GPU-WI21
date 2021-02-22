#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
/*
 * Example kernels for transposing a rectangular host array using a variety of
 * optimizations, including shared memory, unrolling, and memory padding.
 */

// Some kernels assume square blocks
#define BDIMX 16
#define BDIMY BDIMX

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

#define IPAD 1

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void printData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%3.0f ", in[i]);
    }

    printf("\n");
    return;
}

void checkResult(float *hostRef, float *gpuRef, int rows, int cols)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int index = INDEX(i, j, cols);
            if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
                match = 0;
                printf("different on (%d, %d) (offset=%d) element in "
                        "transposed matrix: host %f gpu %f\n", i, j, index,
                        hostRef[index], gpuRef[index]);
                break;
            }
        }
        if (!match) break;
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float *out, float *in, const int nrows, const int ncols)
{
    for (int iy = 0; iy < nrows; ++iy)
    {
        for (int ix = 0; ix < ncols; ++ix)
        {
            out[INDEX(ix, iy, nrows)] = in[INDEX(iy, ix, ncols)];
        }
    }
}

__global__ void copyGmem(float *out, float *in, const int nrows, const int ncols)
{
    // matrix coordinate (ix,iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nrows && col < ncols)
    {
		    // NOTE this is a transpose, not a copy
        out[INDEX(row, col, nrows)] = in[INDEX(row, col, ncols)];
    }
}

__global__ void naiveGmem(float *out, float *in, const int nrows, const int ncols)
{
    // matrix coordinate (ix,iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nrows && col < ncols)
    {
        out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
    }
}

__global__ void naiveGmemUnroll(float *out, float *in, const int nrows,
                                const int ncols)
{
    // Pretend there are twice as many blocks in the x direction
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < nrows)
    {
        if (col < ncols)
        {
            out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
        }

        col += blockDim.x;

        if (col < ncols)
        {
            out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
        }
    }
}

__global__ void transposeSmem(float *out, float *in, int nrows, int ncols)
{
    // static shared memory
    __shared__ float tile[BDIMY][BDIMX];

    // coordinate in original matrix
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);

    if (row < nrows && col < ncols)
    {
      // load data from global memory to shared memory
      tile[threadIdx.y][threadIdx.x] = in[offset];
    }

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

	  // NOTE - need to transpose row and col on block and thread-block level:
	  // 1. swap blocks x-y
	  // 2. swap thread x-y assignment (irow and icol calculations above)
	  // note col still has continuous threadIdx.x -> coalesced gst
	  col = blockIdx.y * blockDim.y + icol;
	  row = blockIdx.x * blockDim.x + irow;

    // linear global memory index for transposed matrix
	  // NOTE nrows is stride of result, row and col are transposed
    unsigned int transposed_offset = INDEX(row, col, nrows);
    // thread synchronization
    __syncthreads();

	  // NOTE invert sizes for write check
    if (row < ncols && col < nrows)
    {
        // store data to global memory from shared memory
        out[transposed_offset] = tile[icol][irow]; // NOTE icol,irow not irow,icol
    }
}

__global__ void transposeSmemUnroll(float *out, float *in, const int nrows, 
                                            const int ncols) 
{
    // static 1D shared memory
    __shared__ float tile[BDIMY][BDIMX * 2];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);
    unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

    if (row < nrows && col < ncols)
    {
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols)
    {
        out[transposed_offset] = tile[irow][icol];
    }
    if (row2 < nrows && col2 < ncols)
    {
        out[transposed_offset2] = tile[irow][blockDim.x + icol];
    }
}

__global__ void transposeSmemUnrollPad(float *out, float *in, const int nrows,
                                       const int ncols)
{
    // static 1D shared memory with padding
    __shared__ float tile[BDIMY][BDIMX * 2 + IPAD];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);
    unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

    if (row < nrows && col < ncols)
    {
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols)
    {
        out[transposed_offset] = tile[irow][icol];
    }
    if (row2 < nrows && col2 < ncols)
    {
        out[transposed_offset2] = tile[irow][blockDim.x + icol];
    }
}

__global__ void transposeSmemUnrollPadDyn (float *out, float *in, const int nrows,
        const int ncols)
{
    // dynamic shared memory
    extern __shared__ float tile[];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);
    unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

    if (row < nrows && col < ncols)
    {
        tile[INDEX(threadIdx.y, threadIdx.x, BDIMX * 2 + IPAD)] = in[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[INDEX(threadIdx.y, blockDim.x + threadIdx.x, BDIMX * 2 + IPAD)] =
            in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols)
    {
        out[transposed_offset] = tile[INDEX(irow, icol, BDIMX * 2 + IPAD)];
    }
    if (row2 < nrows && col2 < ncols)
    {
        out[transposed_offset2] = tile[INDEX(irow, blockDim.x + icol, BDIMX * 2 + IPAD)];
    }
}

__global__ void transposeSmemPad(float *out, float *in, int nrows, int ncols)
{
	// static shared memory
	__shared__ float tile[BDIMY][BDIMX+1];

	// coordinate in original matrix
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);

	if (row < nrows && col < ncols)
	{
		// load data from global memory to shared memory
		tile[threadIdx.y][threadIdx.x] = in[offset];
	}

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// NOTE - need to transpose row and col on block and thread-block level:
	// 1. swap blocks x-y
	// 2. swap thread x-y assignment (irow and icol calculations above)
	// note col still has continuous threadIdx.x -> coalesced gst
	col = blockIdx.y * blockDim.y + icol;
	row = blockIdx.x * blockDim.x + irow;

	// linear global memory index for transposed matrix
	// NOTE nrows is stride of result, row and col are transposed
	unsigned int transposed_offset = INDEX(row, col, nrows);
	// thread synchronization
	__syncthreads();

	// NOTE invert sizes for write check
	if (row < ncols && col < nrows)
	{
		// store data to global memory from shared memory
		out[transposed_offset] = tile[icol][irow]; // NOTE icol,irow not irow,icol
	}
}

__global__ void transposeSmemPadnc(float *out, float *in, int nrows, int ncols)
{
	// static shared memory
	int stride = 33;
	__shared__ float tile[8][32+2];

	// coordinate in original matrix
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	// linear global memory index for original matrix
	unsigned int offset = INDEX(row, col, ncols);

	unsigned int shx = (threadIdx.y & 0x1)*blockDim.x + threadIdx.x;
	unsigned int shy = threadIdx.y >> 1;

	if (row < nrows && col < ncols)
	{
		// load data from global memory to shared memory
		tile[shy][shx] = in[offset];
	}

	// thread index in transposed block
	unsigned int bidx, irow, icol;
	bidx = threadIdx.y * blockDim.x + threadIdx.x;
	irow = bidx / blockDim.y;
	icol = bidx % blockDim.y;

	// NOTE - need to transpose row and col on block and thread-block level:
	// 1. swap blocks x-y
	// 2. swap thread x-y assignment (irow and icol calculations above)
	// note col still has continuous threadIdx.x -> coalesced gst
	col = blockIdx.y * blockDim.y + icol;
	row = blockIdx.x * blockDim.x + irow;

	// linear global memory index for transposed matrix
	// NOTE nrows is stride of result, row and col are transposed
	unsigned int transposed_offset = INDEX(row, col, nrows);
	// thread synchronization
	__syncthreads();

	// NOTE invert sizes for write check
	if (row < ncols && col < nrows)
	{
		// store data to global memory from shared memory
		out[transposed_offset] = tile[icol>>1][(icol& 0x1)*blockDim.y + irow]; // NOTE icol,irow not irow,icol
	}
}

//__global__ void transposeSmemPad(float *out, float *in, int nrows, int ncols)
//{
//    // static shared memory with padding
//    __shared__ float tile[BDIMY][BDIMX + IPAD];
//
//    // coordinate in original matrix
//    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
//    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
//
//    // linear global memory index for original matrix
//    unsigned int offset = INDEX(row, col, ncols);
//
//    // thread index in transposed block
//    unsigned int bidx, irow, icol;
//    bidx = threadIdx.y * blockDim.x + threadIdx.x;
//    irow = bidx / blockDim.y;
//    icol = bidx % blockDim.y;
//
//    // linear global memory index for transposed matrix
//    unsigned int transposed_offset = INDEX(col, row, nrows);
//
//    // transpose with boundary test
//    if (row < nrows && col < ncols)
//    {
//        // load data from global memory to shared memory
//        tile[threadIdx.y][threadIdx.x] = in[offset];
//
//        // thread synchronization
//        __syncthreads();
//
//        // store data to global memory from shared memory
//        out[transposed_offset] = tile[irow][icol];
//    }
//}

__global__ void transposeSmemDyn(float *out, float *in, int nrows, int ncols)
{
    // dynamic shared memory
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);

    // thread index in transposed block
    unsigned int row_idx, col_idx, irow, icol;
    row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    irow    = row_idx / blockDim.y;
    icol    = row_idx % blockDim.y;
    col_idx = irow * blockDim.x + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);

    // transpose with boundary test
    if (row < nrows && col < ncols)
    {
        // load data from global memory to shared memory
        tile[row_idx] = in[offset];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[transposed_offset] = tile[col_idx];
    }
}

__global__ void transposeSmemPadDyn(float *out, float *in, int nrows, int ncols)
{
    // static shared memory with padding
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);

    // thread index in transposed block
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int irow    = idx / blockDim.y;
    unsigned int icol    = idx % blockDim.y;
    unsigned int col_idx = irow * (blockDim.x + IPAD) + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);

    // transpose with boundary test
    if (row < nrows && col < ncols)
    {
        // load data from global memory to shared memory
        tile[row_idx] = in[offset];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[transposed_offset] = tile[col_idx];
    }
}

#define NUM_REPS  100

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
	//StopWatchInterface *timer = NULL;
	cudaEvent_t start, stop;
	float elapsedTimeInMs = 0.0f;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
	//sdkCreateTimer(&timer);
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

    bool iprint = 0;

    // set up array size 2048
    int nrows = 1 << 12;
    int ncols = 1 << 12;

    if (argc > 1) iprint = atoi(argv[1]);

    if (argc > 2) nrows = atoi(argv[2]);

    if (argc > 3) ncols = atoi(argv[3]);

    printf(" with matrix nrows %d ncols %d\n", nrows, ncols);
    size_t ncells = nrows * ncols;
    size_t nBytes = ncells * sizeof(float);

    // execution configuration
    dim3 block (BDIMX, BDIMY);
    /*
     * Map CUDA blocks/threads to output space. Map rows in output to same
     * x-value in CUDA, columns to same y-value.
     */
    dim3 grid ((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
    dim3 grid2 ((grid.x + 2 - 1) / 2, grid.y);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nrows * ncols);

    //  transpose at host side
    transposeHost(hostRef, h_A, nrows, ncols);

    // allocate device memory
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	//sdkStartTimer(&timer);
	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < NUM_REPS; i++)
		copyGmem<<<grid, block >>>(d_C, d_A, nrows, ncols);

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaDeviceSynchronize());
	//get the total elapsed time in ms
	//sdkStopTimer(&timer);
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, nrows * ncols);

    float ibnd = 1000*2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) /
		elapsedTimeInMs;
    ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / (elapsedTimeInMs/ NUM_REPS);
    printf("copyGmem elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", elapsedTimeInMs, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < NUM_REPS; i++)
		naiveGmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
    CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
    ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
    ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / (elapsedTimeInMs / NUM_REPS);
    printf("naiveGmem elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", elapsedTimeInMs, grid.x, grid.y, block.x,
           block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
    naiveGmemUnroll<<<grid2, block>>>(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
    CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
    ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
    ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / elapsedTimeInMs;
    printf("naiveGmemUnroll elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
           "effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
           block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
	
	for (int i = 0; i < NUM_REPS; i++)
    transposeSmem<<<grid, block>>>(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
    CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / (elapsedTimeInMs / NUM_REPS);
	printf("smem elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < NUM_REPS; i++)
		transposeSmemPad<<<grid, block>>>(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
    CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / (elapsedTimeInMs / NUM_REPS);
	printf("smem pad elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

	// tranpose smem pad (nc)
	CHECK(cudaMemset(d_C, 0, nBytes));
	memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < NUM_REPS; i++)
		transposeSmemPadnc << <grid, block >> >(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
	CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / (elapsedTimeInMs / NUM_REPS);
	printf("smem pad (nc) elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
    transposeSmemDyn<<<grid, block, BDIMX*BDIMY*sizeof(float)>>>(d_C, d_A, nrows,
            ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
    CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / elapsedTimeInMs;
	printf("transposeSmemDyn elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
    transposeSmemPadDyn<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(float)>>>(
          d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
	CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / elapsedTimeInMs;
	printf("transposeSmemPadDyn elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
    transposeSmemUnroll<<<grid2, block>>>(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
	CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / elapsedTimeInMs;
	printf("transposeSmemUnroll elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
    transposeSmemUnrollPad<<<grid2, block>>>(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
	CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / elapsedTimeInMs;
	printf("transposeSmemUnrollPad elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	checkCudaErrors(cudaEventRecord(start, 0));
    transposeSmemUnrollPadDyn<<<grid2, block, (BDIMX * 2 + IPAD) * BDIMY *
        sizeof(float)>>>(d_C, d_A, nrows, ncols);
	checkCudaErrors(cudaEventRecord(stop, 0));
	CHECK(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);
	ibnd = 1000 * 2 * ncells * sizeof(float) / (1024.0 * 1024.0 * 1024.0) / elapsedTimeInMs;
	ibnd = 1000 * 2 * ncells * sizeof(float) / 1e9 / elapsedTimeInMs;
	printf("transposeSmemUnrollPadDyn elapsed %f msec <<< grid (%d,%d) block (%d,%d)>>> "
		"effective bandwidth %f GB\n", elapsedTimeInMs, grid2.x, grid2.y, block.x,
		block.y, ibnd);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
