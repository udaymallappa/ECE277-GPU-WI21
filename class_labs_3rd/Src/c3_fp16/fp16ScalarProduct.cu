/**
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// modified for sm < sm5.3

#include "cuda_fp16.h"
#include "helper_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__ void scalarProductKernel(
        half2 const * const a,
        half2 const * const b,
        float * const results,
        size_t const size
        )
{
    const int stride = gridDim.x*blockDim.x;
    __shared__ half2 shArray[128];

    shArray[threadIdx.x] = __float2half2_rn(0.f);
    half2 value = __float2half2_rn(0.f);

	float af, bf, cf;

    for (int i = threadIdx.x + blockDim.x + blockIdx.x; i < size; i+=stride)
    {
		half2 tmpa = a[i];
		half2 tmpb = b[i];

		af = __half2float(tmpa.x);
		bf = __half2float(tmpa.x);
		cf = af*bf;
		value.x = __float2half(cf);

		af = __half2float(tmpa.y);
		bf = __half2float(tmpb.y);
		cf = af*bf;
		value.y = __float2half(cf);
    }

    shArray[threadIdx.x] = value;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        half2 result = shArray[0];
        float f_result = __low2float(result) + __high2float(result);
        results[blockIdx.x] = f_result;
    }
}

void generateInput(half2 * a, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        unsigned temp = rand();
        temp &= 0x83FF83FF;
        temp |= 0x3C003C00;
        a[i] = *(half2*)&temp;
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    const int blocks = 128;
    const int threads = 128;
    size_t size = blocks*threads*16;

    half2 * vec[2];
    half2 * devVec[2];

    float * results;
    float * devResults;

    int devID = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp devProp;
    checkCudaErrors(cudaGetDeviceProperties(&devProp, devID));

    for (int i = 0; i < 2; ++i)
    {
        checkCudaErrors(cudaMallocHost((void**)&vec[i], size*sizeof*vec[i]));
        checkCudaErrors(cudaMalloc((void**)&devVec[i], size*sizeof*devVec[i]));
    }

    checkCudaErrors(cudaMallocHost((void**)&results, blocks*sizeof*results));
    checkCudaErrors(cudaMalloc((void**)&devResults, blocks*sizeof*devResults));

    for (int i = 0; i < 2; ++i)
    {
        generateInput(vec[i], size);
        checkCudaErrors(cudaMemcpy(devVec[i], vec[i], size*sizeof*vec[i], cudaMemcpyHostToDevice));
    }

    scalarProductKernel<<<blocks, threads>>>(devVec[0], devVec[1], devResults, size);

    checkCudaErrors(cudaMemcpy(results, devResults, blocks*sizeof*results, cudaMemcpyDeviceToHost));

    float result = 0;
    for (int i = 0; i < blocks; ++i)
    {
        result += results[i];
    }
    printf("Result: %f \n", result);

    for (int i = 0; i < 2; ++i)
    {
        checkCudaErrors(cudaFree(devVec[i]));
        checkCudaErrors(cudaFreeHost(vec[i]));
    }

    checkCudaErrors(cudaFree(devResults));
    checkCudaErrors(cudaFreeHost(results));

    return EXIT_SUCCESS;
}
