#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <stdio.h>


void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx,
                     const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("%d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("PASS\n\n");
    else
        printf("FAIL\n\n");

}


__global__ void sumMatrixOnGPU(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
  // Program kernel codes properly, otherwise your system could crash /* FIXME */
	// int ix, iy, idx; 
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = iy * nx + ix;

	if ((ix < nx) && (iy < ny)) {
		MatC[idx] = MatA[idx] + MatB[idx];
	}

}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    checkCudaErrors(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    checkCudaErrors(cudaMalloc((void **)&d_MatA, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatB, nBytes));
    checkCudaErrors(cudaMalloc((void **)&d_MatC, nBytes));

    // transfer data from host to device
    checkCudaErrors(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	// chose dimentions of block and grid. 
	dim3 block(32, 32);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    sumMatrixOnGPU<<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);

    // checkCudaErrors kernel error
    checkCudaErrors(cudaGetLastError());

    // copy kernel result back to host side
    checkCudaErrors(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // checkCudaErrors device results
    checkResult(hostRef, gpuRef, nxy);

	  printf("print your name and student id\n");
	  printf(">> Yifan Wang, A53298382 \n\n");

    // free device global memory
    checkCudaErrors(cudaFree(d_MatA));
    checkCudaErrors(cudaFree(d_MatB));
    checkCudaErrors(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    checkCudaErrors(cudaDeviceReset());

    return (0);
}
