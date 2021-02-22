
#include <cuda_runtime.h>
#include <stdio.h>

__device__ int device_C;

__global__ void addkernel(int A, int B)
{
	device_C = A + B;
}

int main(int argc, char **argv)
{
	int host_C;

	addkernel << <1, 1 >> > (2, 3);

	cudaMemcpyFromSymbol(&host_C, device_C, sizeof(int), 0, cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_C, C, sizeof(int), cudaMemcpyDeviceToHost);

	printf("C=%d\n", host_C);
}

