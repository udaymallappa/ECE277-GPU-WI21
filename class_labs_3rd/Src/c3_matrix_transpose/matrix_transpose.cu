#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

// Some kernels assume square blocks
#define BDIMX 16
#define BDIMY 16

__global__ void transposeNaiveRow(float *out, float *in, const int nrows, const int ncols)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (iy < nrows && ix < ncols) {
		out[ix*nrows + iy] = in[iy*ncols + ix];
	}
}

__global__ void transposeNaiveCol(float *out, float *in, const int nrows, const int ncols)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (iy < nrows && ix < ncols) {
		out[iy*ncols + ix] = in[ix*nrows + iy];
	}
}

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))


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

	if (match)
		printf("PASS\n\n");
	else
		printf("FAIL\n\n");
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

int main(int argc, char **argv)
{
    bool iprint = 0;

    int nrows = 1 << 10;
    int ncols = 1 << 10;

    printf(" with matrix nrows %d ncols %d\n", nrows, ncols);
    size_t ncells = nrows * ncols;
    size_t nBytes = ncells * sizeof(float);

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
    checkCudaErrors(cudaMalloc((float**)&d_A, nBytes));
    checkCudaErrors(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

	dim3 block(BDIMX, BDIMY);
	dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);

	printf("Transpose: row-wise read, column-wise write\n");
	transposeNaiveRow <<<grid, block>>>(d_C, d_A, nrows, ncols);

    checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprint) printData(gpuRef, ncells);

    checkResult(hostRef, gpuRef, ncols, nrows);

	printf("Transpose: column-wise read, row-wise write\n");
	transposeNaiveCol << <grid, block >> >(d_C, d_A, nrows, ncols);

	checkCudaErrors(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	if (iprint) printData(gpuRef, ncells);

	checkResult(hostRef, gpuRef, ncols, nrows);

    // free host and device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);
    return EXIT_SUCCESS;
}
