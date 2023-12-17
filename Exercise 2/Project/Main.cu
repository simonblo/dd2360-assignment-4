#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BUFFER  1048576
#define SEGMENT 4096
#define STREAM  true

__global__ void gpuVectorAdd(float* bufferIn1, float* bufferIn2, float* bufferOut, int bufferSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < bufferSize) bufferOut[tid] = bufferIn1[tid] + bufferIn2[tid];
}

__global__ void gpuVectorAdd(float* bufferIn1, float* bufferIn2, float* bufferOut, int bufferSize, int bufferOffset)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + bufferOffset;
	if (tid < bufferSize) bufferOut[tid] = bufferIn1[tid] + bufferIn2[tid];
}

void VectorAdd(float* cpuBufferIn1, float* cpuBufferIn2, float* cpuBufferOut, float* gpuBufferIn1, float* gpuBufferIn2, float* gpuBufferOut)
{
	cudaMemcpy(gpuBufferIn1, cpuBufferIn1, BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuBufferIn2, cpuBufferIn2, BUFFER * sizeof(float), cudaMemcpyHostToDevice);

	int threads = 64;
	int blocks  = (BUFFER + threads - 1) / threads;

	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, BUFFER);

	cudaDeviceSynchronize();

	cudaMemcpy(cpuBufferOut, gpuBufferOut, BUFFER * sizeof(float), cudaMemcpyHostToDevice);
}

void VectorAdd(float* cpuBufferIn1, float* cpuBufferIn2, float* cpuBufferOut, float* gpuBufferIn1, float* gpuBufferIn2, float* gpuBufferOut, cudaStream_t* stream)
{
	int stride = BUFFER / 4;

	cudaMemcpyAsync(&gpuBufferIn1[0 * stride], &cpuBufferIn1[0 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(&gpuBufferIn1[1 * stride], &cpuBufferIn1[1 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(&gpuBufferIn1[2 * stride], &cpuBufferIn1[2 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[2]);
	cudaMemcpyAsync(&gpuBufferIn1[3 * stride], &cpuBufferIn1[3 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[3]);

	cudaMemcpyAsync(&gpuBufferIn2[0 * stride], &cpuBufferIn2[0 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(&gpuBufferIn2[1 * stride], &cpuBufferIn2[1 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(&gpuBufferIn2[2 * stride], &cpuBufferIn2[2 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[2]);
	cudaMemcpyAsync(&gpuBufferIn2[3 * stride], &cpuBufferIn2[3 * stride], stride * sizeof(float), cudaMemcpyHostToDevice, stream[3]);

	int threads = 64;
	int blocks  = (stride + threads - 1) / threads;

	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, BUFFER, 0 * stride);
	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, BUFFER, 1 * stride);
	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, BUFFER, 2 * stride);
	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, BUFFER, 3 * stride);

	cudaMemcpyAsync(&cpuBufferOut[0 * stride], &gpuBufferOut[0 * stride], stride * sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
	cudaMemcpyAsync(&cpuBufferOut[1 * stride], &gpuBufferOut[1 * stride], stride * sizeof(float), cudaMemcpyDeviceToHost, stream[1]);
	cudaMemcpyAsync(&cpuBufferOut[2 * stride], &gpuBufferOut[2 * stride], stride * sizeof(float), cudaMemcpyDeviceToHost, stream[2]);
	cudaMemcpyAsync(&cpuBufferOut[3 * stride], &gpuBufferOut[3 * stride], stride * sizeof(float), cudaMemcpyDeviceToHost, stream[3]);

	cudaDeviceSynchronize();
}

double GetSeconds()
{
	struct timespec tp;
	timespec_get(&tp, TIME_UTC);
	return ((double)tp.tv_sec + (double)tp.tv_nsec * 1.e-9);
}

int main()
{
	float* cpuBufferIn1;
	float* cpuBufferIn2;
	float* cpuBufferOut;
	float* gpuBufferIn1;
	float* gpuBufferIn2;
	float* gpuBufferOut;

	cudaStream_t stream[4];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	cudaStreamCreate(&stream[2]);
	cudaStreamCreate(&stream[3]);

	cudaHostAlloc((void**)&cpuBufferIn1, BUFFER * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cpuBufferIn2, BUFFER * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cpuBufferOut, BUFFER * sizeof(float), cudaHostAllocDefault);

	cudaMalloc((void**)&gpuBufferIn1, BUFFER * sizeof(float));
	cudaMalloc((void**)&gpuBufferIn2, BUFFER * sizeof(float));
	cudaMalloc((void**)&gpuBufferOut, BUFFER * sizeof(float));

	srand(time(NULL));

	for (int i = 0; i != BUFFER; ++i)
	{
		cpuBufferIn1[i] = (float)rand() / (float)RAND_MAX;
		cpuBufferIn2[i] = (float)rand() / (float)RAND_MAX;
	}

	double time0 = GetSeconds();
	#if STREAM
	VectorAdd(cpuBufferIn1, cpuBufferIn2, cpuBufferOut, gpuBufferIn1, gpuBufferIn2, gpuBufferOut, stream);
	#else
	VectorAdd(cpuBufferIn1, cpuBufferIn2, cpuBufferOut, gpuBufferIn1, gpuBufferIn2, gpuBufferOut);
	#endif
	double time1 = GetSeconds();

	int errorCount = 0;

	for (int i = 0; i != BUFFER; ++i)
	{
		errorCount += ((cpuBufferIn1[i] + cpuBufferIn2[i]) != cpuBufferOut[i]);
	}

	printf("Elements: %d\n", BUFFER);
	printf("Errors:   %d\n", errorCount);
	printf("Time:     %f\n", time1 - time0);

	cudaFree(gpuBufferIn1);
	cudaFree(gpuBufferIn2);
	cudaFree(gpuBufferOut);

	cudaFreeHost(cpuBufferIn1);
	cudaFreeHost(cpuBufferIn2);
	cudaFreeHost(cpuBufferOut);

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaStreamDestroy(stream[2]);
	cudaStreamDestroy(stream[3]);

	return 0;
}