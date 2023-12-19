#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BUFFER 8388608
#define STRIDE 1048576
#define STREAM true

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
	int threads = 64;
	int blocks  = (BUFFER + threads - 1) / threads;

	cudaMemcpy(gpuBufferIn1, cpuBufferIn1, BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuBufferIn2, cpuBufferIn2, BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	gpuVectorAdd<<<blocks, threads>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, BUFFER);
	cudaDeviceSynchronize();
	cudaMemcpy(cpuBufferOut, gpuBufferOut, BUFFER * sizeof(float), cudaMemcpyDeviceToHost);
}

void VectorAdd(float* cpuBufferIn1, float* cpuBufferIn2, float* cpuBufferOut, float* gpuBufferIn1, float* gpuBufferIn2, float* gpuBufferOut, cudaStream_t* stream)
{
	int threads = 64;
	int blocks  = (STRIDE + threads - 1) / threads;

	// For some reason the first kernel launch takes a massive amount of CPU time which seems to
	// scale with the input size. This slows everything down since the GPU manages to complete all
	// memory transfers before the CPU manages to complete the first kernel launch which causes the
	// GPU to sit idle. We can make a small "dummy" launch at the start to minimize this latency.
	gpuVectorAdd<<<1, 32>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, 0, 0);

	for (int i = 0; i != BUFFER / STRIDE; ++i)
	{
		cudaMemcpyAsync(&gpuBufferIn1[i * STRIDE], &cpuBufferIn1[i * STRIDE], STRIDE * sizeof(float), cudaMemcpyHostToDevice, stream[i % 4]);
		cudaMemcpyAsync(&gpuBufferIn2[i * STRIDE], &cpuBufferIn2[i * STRIDE], STRIDE * sizeof(float), cudaMemcpyHostToDevice, stream[i % 4]);
		gpuVectorAdd<<<blocks, threads, 0, stream[i % 4]>>>(gpuBufferIn1, gpuBufferIn2, gpuBufferOut, BUFFER, i * STRIDE);
		cudaMemcpyAsync(&cpuBufferOut[i * STRIDE], &gpuBufferOut[i * STRIDE], STRIDE * sizeof(float), cudaMemcpyDeviceToHost, stream[i % 4]);
	}

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