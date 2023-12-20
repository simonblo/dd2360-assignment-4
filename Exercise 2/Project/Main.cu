#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>

#define BUFFER 8388608
#define STRIDE 1048576
#define STREAM true

__global__ void KernelVectorAdd(float* input1, float* input2, float* output, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size) output[tid] = input1[tid] + input2[tid];
}

__global__ void KernelVectorAdd(float* input1, float* input2, float* output, int size, int offset)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x + offset;
	if (tid < size) output[tid] = input1[tid] + input2[tid];
}

void VectorAdd(float* cpuInput1, float* cpuInput2, float* cpuOutput, float* gpuInput1, float* gpuInput2, float* gpuOutput)
{
	int threads = 64;
	int blocks  = (BUFFER + threads - 1) / threads;

	cudaMemcpy(gpuInput1, cpuInput1, BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuInput2, cpuInput2, BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	KernelVectorAdd<<<blocks, threads>>>(gpuInput1, gpuInput2, gpuOutput, BUFFER);
	cudaDeviceSynchronize();
	cudaMemcpy(cpuOutput, gpuOutput, BUFFER * sizeof(float), cudaMemcpyDeviceToHost);
}

void VectorAdd(float* cpuInput1, float* cpuInput2, float* cpuOutput, float* gpuInput1, float* gpuInput2, float* gpuOutput, cudaStream_t* stream)
{
	int threads = 64;
	int blocks  = (STRIDE + threads - 1) / threads;

	// For some reason the first kernel launch takes a massive amount of CPU time which seems to
	// scale with the input size. This slows everything down since the GPU manages to complete all
	// memory transfers before the CPU manages to complete the first kernel launch which causes the
	// GPU to sit idle. We can make a small "dummy" launch at the start to minimize this latency.
	KernelVectorAdd<<<1, 32>>>(gpuInput1, gpuInput2, gpuOutput, 0, 0);

	for (int i = 0; i != BUFFER / STRIDE; ++i)
	{
		cudaMemcpyAsync(&gpuInput1[i * STRIDE], &cpuInput1[i * STRIDE], STRIDE * sizeof(float), cudaMemcpyHostToDevice, stream[i % 4]);
		cudaMemcpyAsync(&gpuInput2[i * STRIDE], &cpuInput2[i * STRIDE], STRIDE * sizeof(float), cudaMemcpyHostToDevice, stream[i % 4]);
		KernelVectorAdd<<<blocks, threads, 0, stream[i % 4]>>>(gpuInput1, gpuInput2, gpuOutput, BUFFER, i * STRIDE);
		cudaMemcpyAsync(&cpuOutput[i * STRIDE], &gpuOutput[i * STRIDE], STRIDE * sizeof(float), cudaMemcpyDeviceToHost, stream[i % 4]);
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
	float* cpuInput1;
	float* cpuInput2;
	float* cpuOutput;
	float* gpuInput1;
	float* gpuInput2;
	float* gpuOutput;

	cudaStream_t stream[4];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	cudaStreamCreate(&stream[2]);
	cudaStreamCreate(&stream[3]);

	cudaHostAlloc((void**)&cpuInput1, BUFFER * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cpuInput2, BUFFER * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cpuOutput, BUFFER * sizeof(float), cudaHostAllocDefault);

	cudaMalloc((void**)&gpuInput1, BUFFER * sizeof(float));
	cudaMalloc((void**)&gpuInput2, BUFFER * sizeof(float));
	cudaMalloc((void**)&gpuOutput, BUFFER * sizeof(float));

	srand(time(NULL));

	for (int i = 0; i != BUFFER; ++i)
	{
		cpuInput1[i] = (float)rand() / (float)RAND_MAX;
		cpuInput2[i] = (float)rand() / (float)RAND_MAX;
	}

	double time0 = GetSeconds();
	#if STREAM
	VectorAdd(cpuInput1, cpuInput2, cpuOutput, gpuInput1, gpuInput2, gpuOutput, stream);
	#else
	VectorAdd(cpuInput1, cpuInput2, cpuOutput, gpuInput1, gpuInput2, gpuOutput);
	#endif
	double time1 = GetSeconds();

	int errorCount = 0;

	for (int i = 0; i != BUFFER; ++i)
	{
		errorCount += ((cpuInput1[i] + cpuInput2[i]) != cpuOutput[i]);
	}

	printf("Elements: %d\n", BUFFER);
	printf("Errors:   %d\n", errorCount);
	printf("Time:     %f\n", time1 - time0);

	cudaFree(gpuInput1);
	cudaFree(gpuInput2);
	cudaFree(gpuOutput);

	cudaFreeHost(cpuInput1);
	cudaFreeHost(cpuInput2);
	cudaFreeHost(cpuOutput);

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaStreamDestroy(stream[2]);
	cudaStreamDestroy(stream[3]);

	return 0;
}