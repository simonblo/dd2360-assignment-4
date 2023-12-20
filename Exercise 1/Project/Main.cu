#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define X 800
#define Y 600

__global__ void KernelProcessPicture(float* input, float* output, int x, int y)
{
	uint2 tid;
	tid.x = threadIdx.x + blockIdx.x * blockDim.x;
	tid.y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tid.x < x) && (tid.y < y))
	{
		output[tid.x + tid.y * x] = 2.0f * input[tid.x + tid.y * x];
	}
}

int main()
{
	float* cpuInput;
	float* cpuOutput;
	float* gpuInput;
	float* gpuOutput;

	srand(time(NULL));

	cudaHostAlloc((void**)&cpuInput,  X * Y * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cpuOutput, X * Y * sizeof(float), cudaHostAllocDefault);

	cudaMalloc((void**)&gpuInput,  X * Y * sizeof(float));
	cudaMalloc((void**)&gpuOutput, X * Y * sizeof(float));

	for (int i = 0; i != X * Y; ++i)
	{
		cpuInput[i] = (float)rand() / (float)RAND_MAX;
	}

	cudaMemcpy(gpuInput, cpuInput, X * Y * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads;
	threads.x = 16;
	threads.y = 16;
	threads.z = 1;

	dim3 blocks;
	blocks.x = (X + threads.x - 1) / threads.x;
	blocks.y = (Y + threads.y - 1) / threads.y;
	blocks.z = 1;

	KernelProcessPicture<<<blocks, threads>>>(gpuInput, gpuOutput, X, Y);
	cudaDeviceSynchronize();

	cudaMemcpy(cpuOutput, gpuOutput, X * Y * sizeof(float), cudaMemcpyDeviceToHost);

	printf("X: %u\n", X);
	printf("Y: %u\n", Y);

	cudaFree(gpuInput);
	cudaFree(gpuOutput);

	cudaFreeHost(cpuInput);
	cudaFreeHost(cpuOutput);

	return 0;
}