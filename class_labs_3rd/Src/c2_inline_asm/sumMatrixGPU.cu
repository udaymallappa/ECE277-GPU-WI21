#include <cuda_runtime.h>
#include <stdio.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>  
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void initialData(int2 *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i].x = (int)(rand() * 100);
		ip[i].y = (int)(rand() * 100);
    }

    return;
}

void sumMatrixOnHost(int2 *A, int2 *B, int2 *C, const int nx, const int ny)
{
	int2 *ia = A;
	int2 *ib = B;
	int2 *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix].x = ia[ix].x + ib[ix].x;
			ic[ix].y = ia[ix].y + ib[ix].y;
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}


void checkResult(int2 *hostRef, int2 *gpuRef, const int N)
{
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if ((hostRef[i].x != gpuRef[i].x) || (hostRef[i].y != gpuRef[i].y))
        {
            match = 0;
            printf("host (%d %d) gpu (%d %d)\n", hostRef[i].x, hostRef[i].y, gpuRef[i].x, gpuRef[i].y);
            break;
        }
    }

    if (!match)
        printf("Arrays do not match.\n\n");
	else
		printf("No Error.\n\n");
}

// grid 2D block 2D
__global__ void sumMatrixGPU(int2 *MatA, int2 *MatB, int2 *MatC, int nx,
                             int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny) {
		MatC[idx].x = MatA[idx].x + MatB[idx].x;
		MatC[idx].y = MatA[idx].y + MatB[idx].y;
	}
}

__global__ void sumMatrixGPU_inlineASM(int2 *MatA, int2 *MatB, int2 *MatC, int nx,
	int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny) {
		MatC[idx].x = 0;
		MatC[idx].y = 0;

		int2 a, b, c;

		asm volatile("ld.cg.v2.s32" " {%0,%1}, [%2];": "=r"(a.x), "=r"(a.y) : "l"(MatA + idx));
		asm volatile("ld.cg.v2.s32" " {%0,%1}, [%2];": "=r"(b.x), "=r"(b.y) : "l"(MatB + idx));

		c.x = a.x + b.x;
		c.y = a.y + b.y;

		asm volatile("st.cg.v2.s32 [%0], {%1,%2};"::"l"(MatC + idx), "r"(c.x), "r"(c.y));
		//MatC[idx].x = MatA[idx].x + MatB[idx].x;

	}
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);
    // set up data size of matrix
    int nx, ny;
    int ishift = 2;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int2);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    int2 *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (int2 *)malloc(nBytes);
    h_B = (int2 *)malloc(nBytes);
    hostRef = (int2 *)malloc(nBytes);
    gpuRef = (int2 *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    // add matrix at host side for result checks
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    // malloc device global memory
    int2 *d_MatA, *d_MatB, *d_MatC;
	checkCudaErrors(cudaMalloc((void **)&d_MatA, nBytes));
	checkCudaErrors(cudaMalloc((void **)&d_MatB, nBytes));
	checkCudaErrors(cudaMalloc((void **)&d_MatC, nBytes));

    // invoke kernel at host side
    int dimx = 4;
    int dimy = 4;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // transfer data from host to device
	checkCudaErrors(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    //sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
 
	sumMatrixGPU_inlineASM << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
    printf("sumMatrix on gpu :\t  <<<(%d,%d), (%d,%d)>>> \n", 
           grid.x, grid.y, block.x, block.y);

	checkCudaErrors(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check kernel error
	checkCudaErrors(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
	checkCudaErrors(cudaFree(d_MatA));
	checkCudaErrors(cudaFree(d_MatB));
	checkCudaErrors(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return (0);
}
