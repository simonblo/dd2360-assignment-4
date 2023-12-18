#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <time.h>

#define gpuCheck(statement)                                        \
{                                                                  \
	cudaError_t error = statement;                                 \
	if (error != cudaSuccess)                                      \
	{                                                              \
		printf("ERROR. Failed to run statement %s\n", #statement); \
	}                                                              \
}

#define cublasCheck(statement)                                            \
{                                                                         \
	cublasStatus_t error = statement;                                     \
	if (error != CUBLAS_STATUS_SUCCESS)                                   \
	{                                                                     \
		printf("ERROR. Failed to run cuBLAS statement %s\n", #statement); \
	}                                                                     \
}

#define cusparseCheck(statement)                                       \
{                                                                      \
	cusparseStatus_t error = statement;                                \
	if (error != CUSPARSE_STATUS_SUCCESS)                              \
	{                                                                  \
		printf("ERROR. Failed to run cuSPARSE stmt %s\n", #statement); \
	}                                                                  \
}

struct timespec timerStart;
struct timespec timerStop;

void cpuTimerStart()
{
	timespec_get(&timerStart, TIME_UTC);
}

void cpuTimerStop(const char* info)
{
	timespec_get(&timerStop, TIME_UTC);
	double time = 1000000000.0 * (timerStop.tv_sec - timerStart.tv_sec) + (timerStop.tv_nsec - timerStart.tv_nsec);
	printf("Timing - %s. Elapsed %.0f nanoseconds \n", info, time);
}

// Initialize the sparse matrix needed for the heat time step.
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX, double alpha)
{
	// Stencil from the finete difference discretization of the equation.
	double stencil[] = { 1, -2, 1 };

	// Variable holding the position to insert a new element.
	size_t ptr = 0;

	// Insert a row of zeros at the beginning of the matrix.
	ArowPtr[1] = ptr;

	// Fill the non zero entries of the matrix.
	for (int i = 1; i < (dimX - 1); ++i)
	{
		// Insert the elements: A[i][i-1], A[i][i], A[i][i+1].
		for (int k = 0; k < 3; ++k)
		{
			// Set the value for A[i][i+k-1].
			A[ptr] = stencil[k];

			// Set the column index for A[i][i+k-1].
			AcolIndx[ptr++] = i + k - 1;
		}

		// Set the number of newly added elements.
		ArowPtr[i + 1] = ptr;
	}

	// Insert a row of zeros at the end of the matrix.
	ArowPtr[dimX] = ptr;
}

int main(int argc, char** argv)
{
	// TODO.

	int device = 0;            // Device to be used
	int dimX;                  // Dimension of the metal rod
	int nsteps;                // Number of time steps to perform
	double alpha = 0.4;        // Diffusion coefficient
	double* temp;              // Array to store the final time step
	double* A;                 // Sparse matrix A values in the CSR format
	int* ARowPtr;              // Sparse matrix A row pointers in the CSR format
	int* AColIndx;              // Sparse matrix A col values in the CSR format
	int nzv;                   // Number of non zero values in the sparse matrix
	double* tmp;               // Temporal array of dimX for computations
	size_t bufferSize = 0;     // Buffer size needed by some routines
	void* buffer = nullptr;    // Buffer used by some routines in the libraries
	int concurrentAccessQ;     // Check if concurrent access flag is set
	double zero = 0;           // Zero constant
	double one = 1;            // One constant
	double norm;               // Variable for norm values
	double error;              // Variable for storing the relative error
	double tempLeft = 200.;    // Left heat source applied to the rod
	double tempRight = 300.;   // Right heat source applied to the rod
	cublasHandle_t cublasHandle;      // cuBLAS handle
	cusparseHandle_t cusparseHandle;  // cuSPARSE handle
	cusparseSpMatDescr_t Adescriptor;   // Mat descriptor needed by cuSPARSE
	cusparseDnVecDescr_t Tdescriptor; // TODO: Rename.
	cusparseDnVecDescr_t Ydescriptor; // TODO: Rename.





	// Read the arguments from the command line.
	dimX   = atoi(argv[1]);
	nsteps = atoi(argv[2]);

	// Print input arguments.
	printf("The X dimension of the grid is %d\n",         dimX);
	printf("The number of time steps to perform is %d\n", nsteps);

	// Get if the cudaDevAttrConcurrentManagedAccess flag is set.
	gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

	// Calculate the number of non zero values in the sparse matrix. This number is known from the structure of the sparse matrix.
	nzv = 3 * dimX - 6;
	
	// Allocate the temp, tmp and the sparse matrix arrays using Unified Memory.
	cpuTimerStart();
	gpuCheck(cudaMallocManaged((void**)&temp, dimX * sizeof(double)));
	gpuCheck(cudaMallocManaged((void**)&A, nzv * sizeof(double)));
	gpuCheck(cudaMallocManaged((void**)&ARowPtr, (dimX + 1) * sizeof(int)));
	gpuCheck(cudaMallocManaged((void**)&AColIndx, nzv * sizeof(int)));
	gpuCheck(cudaMallocManaged((void**)&tmp, dimX * sizeof(double)));
	cpuTimerStop("Allocating device memory");

	// Check if concurrentAccessQ is non zero in order to prefetch memory.
	if (concurrentAccessQ)
	{
		// Prefetch in Unified Memory asynchronously to the CPU.
		cpuTimerStart();
		cudaMemPrefetchAsync(temp, dimX * sizeof(double), cudaCpuDeviceId);
		cudaMemPrefetchAsync(A, nzv * sizeof(double), cudaCpuDeviceId);
		cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), cudaCpuDeviceId);
		cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), cudaCpuDeviceId);
		cudaMemPrefetchAsync(tmp, dimX * sizeof(double), cudaCpuDeviceId); 
		cpuTimerStop("Prefetching GPU memory to the host");
	}

	// Initialize the sparse matrix.
	cpuTimerStart();
	matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
	cpuTimerStop("Initializing the sparse matrix on the host");

	// Initiliaze the boundary conditions for the heat equation.
	cpuTimerStart();
	memset(temp, 0, sizeof(double) * dimX);
	temp[0] = tempLeft;
	temp[dimX - 1] = tempRight;
	cpuTimerStop("Initializing memory on the host");

	// Check if concurrentAccessQ is non zero in order to prefetch memory.
	if (concurrentAccessQ)
	{
		// Prefetch in Unified Memory asynchronously to the GPU.
		cpuTimerStart();
		cudaMemPrefetchAsync(temp, dimX * sizeof(double), device);
		cudaMemPrefetchAsync(A, nzv * sizeof(double), device);
		cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), device);
		cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), device);
		cudaMemPrefetchAsync(tmp, dimX * sizeof(double), device);
		cpuTimerStop("Prefetching GPU memory to the device");
	}

	// Create the cuBLAS handle and the cuSPARSE handle.
	cublasCreate_v2(&cublasHandle);
	cusparseCreate(&cusparseHandle);

	// Set the cuBLAS pointer mode to CUSPARSE_POINTER_MODE_HOST.
	cublasSetPointerMode_v2(cublasHandle, (cublasPointerMode_t)CUSPARSE_POINTER_MODE_HOST);

	// Create the sparse matrix descriptor and the dense vector descriptors used by cuSPARSE.
	cusparseCreateCsr(&Adescriptor, dimX, dimX, nzv, ARowPtr, AColIndx, A, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cusparseCreateDnVec(&Tdescriptor, dimX, temp, CUDA_R_64F);
	cusparseCreateDnVec(&Ydescriptor, dimX, tmp, CUDA_R_64F);

	// Get the buffer size needed by the sparse matrix vector (SpMV) CSR routine of cuSPARSE.
	cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, Adescriptor, Tdescriptor, &zero, Ydescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

	// Allocate the working buffer needed by cuSPARSE.
	cudaMalloc(&buffer, bufferSize);

	// Perform the time step iterations.
	for (int i = 0; i != nsteps; ++i)
	{
		// Calculate the sparse matrix vector (SpMV) routine corresponding to tmp = 1 * A * temp + 0 * tmp.
		cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, Adescriptor, Tdescriptor, &zero, Ydescriptor, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

		// Calculate the dense vector scalar (Daxpy) routine corresponding to temp = alpha * tmp + temp.
		cublasDaxpy_v2(cublasHandle, dimX, &alpha, tmp, 1, temp, 1);

		// Calculate the norm of the dense vector corresponding to norm = ||temp||.
		cublasDnrm2_v2(cublasHandle, dimX, temp, 1, &norm);

		// If the norm of A*temp is smaller than 10^-4 exit the loop.
		if (norm < 1e-4) break;
	}

	// Calculate the exact solution using thrust.
	thrust::device_ptr<double> thrustPtr(tmp);
	thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft, (tempRight - tempLeft) / (dimX - 1));

	// Calculate the relative approximation error corresponding to tmp = -1 * temp + tmp.
	one = -1;
	cublasDaxpy_v2(cublasHandle, dimX, &one, temp, 1, tmp, 1);

	// Calculate the norm of the absolute error corresponding to norm = ||tmp||.
	cublasDnrm2_v2(cublasHandle, dimX, tmp, 1, &norm);

	// Calculate the norm of temp corresponding to ||temp||.
	error = norm;
	cublasDnrm2_v2(cublasHandle, dimX, temp, 1, &norm);

	// Calculate and print the relative error.
	error = error / norm;
	printf("The relative error of the approximation is %f\n", error);

	// Destroy the sparse matrix descriptor and the dense vector descriptor used by cuSPARSE.
	cusparseDestroySpMat(Adescriptor);
	cusparseDestroyDnVec(Tdescriptor);
	cusparseDestroyDnVec(Ydescriptor);

	// Destroy the cuSPARSE handle and the cuBLAS handle.
	cusparseDestroy(cusparseHandle);
	cublasDestroy_v2(cublasHandle);

	// Deallocate memory.
	cudaFree(temp);
	cudaFree(A);
	cudaFree(ARowPtr);
	cudaFree(AColIndx);
	cudaFree(tmp);
	cudaFree(buffer);

	return 0;
}