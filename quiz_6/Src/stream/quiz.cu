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

	// GMEM allication gbuf
	cudaMalloc( (void **) &d_a, size);
	cudaMalloc( (void **) &d_b, size);
	cudaMalloc( (void **) &d_c, size);

	a = (int *)malloc( size );
	b = (int *)malloc( size );
	c = (int *)malloc( size );
	golden = (int *)malloc(size);

	for( int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		golden[i] = a[i] + b[i];
		c[i] = 0;
	}

	// cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
	// cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

	// declared streams
	cudaStream_t streams[NSTREAMS];
	int streamSize = size / NSTREAMS; // streamBytes = stream_size * sizeof(float);

	// event start
	// cudaEventRecord(addEvent, 0);

	for (int i = 0; i < NSTREAMS; ++i) {

		cudaStreamCreate(&streams[i]);

		int offset = i * streamSize;

		// cudaMemocpyAsync(cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&d_a[offset], &a[offset], streamSize, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(&d_b[offset], &b[offset], streamSize, cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(&d_c[offset], &c[offset], streamSize, cudaMemcpyHostToDevice, stream[i]);

		int mgrid = (N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK / NSTREAMS;
		int mblock = THREADS_PER_BLOCK / NSTREAMS;
		vector_add <<< mgrid,  mblock, 0, streams[i]>>>( &d_a[offset], &d_b[offset], &d_c[offset] );
		// cudaMemocpyAsync(cudaMemcpyDeviceToHost, streams[i]);
		cudaMemcpyAsync(&a[offset], &d_a[offset], streamSize, cudaMemcpyDeviceToHost, stream[i]);
		cudaMemcpyAsync(&b[offset], &d_b[offset], streamSize, cudaMemcpyDeviceToHost, stream[i]);
		cudaMemcpyAsync(&c[offset], &d_c[offset], streamSize, cudaMemcpyDeviceToHost, stream[i]);

		cudaStreamDestroy(streams[i]);

	}

	// event end
	// cudaEventRecord(addEvent, 0);

	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );

	bool pass = true;
	for (int i = 0; i < N; i++) {
		if (golden[i] != c[i])
			pass = false;
	}
	
	if (pass)
		printf("PASS\n");
	else
		printf("FAIL\n");

	printf("print your name and id\n");

	free(a);
	free(b);
	free(c);
	free(golden);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} 
