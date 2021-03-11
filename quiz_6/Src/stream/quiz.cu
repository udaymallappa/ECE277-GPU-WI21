#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>


// DO NOT change the kernel function
__global__ void vector_add(int *a, int *b, int *c)
{
// DO NOT change the kernel function
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}


#define N (2048*2048)
#define THREADS_PER_BLOCK 128

#define NSTREAMS 4


int main()
{
    int *a, *b, *c, *golden;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof( int );

	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	cudaHostAlloc((void **)&a, size, cudaHostAllocDefault);
	cudaHostAlloc((void **)&b, size, cudaHostAllocDefault);
	cudaHostAlloc((void **)&c, size, cudaHostAllocDefault);

	//a = (int *)malloc( size );
	//b = (int *)malloc( size );
	//c = (int *)malloc( size );

	golden = (int *)malloc(size);

	for( int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		golden[i] = a[i] + b[i];
		c[i] = 0;
	}

	cudaStream_t streams[NSTREAMS];
	for (int i = 0; i < NSTREAMS; ++i) {
		cudaStreamCreate(&streams[i]);
	}

	int nsdata = N / NSTREAMS;
	int iBytes = size / NSTREAMS;

	dim3 mgrid((nsdata + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);
	dim3 mblock(THREADS_PER_BLOCK);

	for (int i = 0; i < NSTREAMS; ++i) {
		int offset = i * nsdata;
		
		cudaMemcpyAsync(&d_a[offset], &a[offset], iBytes, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&d_b[offset], &b[offset], iBytes, cudaMemcpyHostToDevice, streams[i]);

		vector_add << < mgrid, mblock, 0, streams[i] >> > (&d_a[offset], &d_b[offset], &d_c[offset]);
		cudaMemcpyAsync(&c[offset], &d_c[offset], iBytes, cudaMemcpyDeviceToHost, streams[i]);
	}

	for (int i = 0; i < NSTREAMS; ++i) {
		cudaStreamSynchronize(streams[i]);
	}

	for (int i = 0; i < NSTREAMS; ++i) {
		cudaStreamDestroy(streams[i]);
	}

	bool pass = true;
	for (int i = 0; i < N; i++) {
		if (golden[i] != c[i]) {
			pass = false;
			//printf("%i %d %d \n", i, golden[i], c[i]);
		}
	}
	
	if (pass)
		printf("PASS\n");
	else
		printf("FAIL\n");

	printf("print your name and id \n>> Yifan Wang, A53298382 \n\n");

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);

	//free(a);
	//free(b);
	//free(c);

	free(golden);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} 
