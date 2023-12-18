﻿#include <cublas_v2.h>
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

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
          break;                                                     \
      }                                                              \
  } while (0)

struct timespec t_start, t_end;

void cputimer_start()
{
	timespec_get(&t_start, TIME_UTC);
}

void cputimer_stop(const char* info)
{
	timespec_get(&t_end, TIME_UTC);
	double time = (1000000000.0 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_nsec - t_start.tv_nsec);
	printf("Timing - %s. \t\tElapsed %.0f nanoseconds \n", info, time);
}

// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX, double alpha)
{
	// Stencil from the finete difference discretization of the equation
	double stencil[] = { 1, -2, 1 };
	// Variable holding the position to insert a new element
	size_t ptr = 0;
	// Insert a row of zeros at the beginning of the matrix
	ArowPtr[1] = ptr;
	// Fill the non zero entries of the matrix
	for (int i = 1; i < (dimX - 1); ++i) {
		// Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
		for (int k = 0; k < 3; ++k) {
			// Set the value for A[i][i+k-1]
			A[ptr] = stencil[k];
			// Set the column index for A[i][i+k-1]
			AcolIndx[ptr++] = i + k - 1;
		}
		// Set the number of newly added elements
		ArowPtr[i + 1] = ptr;
	}
	// Insert a row of zeros at the end of the matrix
	ArowPtr[dimX] = ptr;
}

int main(int argc, char** argv)
{
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
	cusparseMatDescr_t Adescriptor;   // Mat descriptor needed by cuSPARSE

	// Read the arguments from the command line
	dimX = atoi(argv[1]);
	nsteps = atoi(argv[2]);

	// Print input arguments
	printf("The X dimension of the grid is %d \n", dimX);
	printf("The number of time steps to perform is %d \n", nsteps);

	// Get if the cudaDevAttrConcurrentManagedAccess flag is set
	gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

	// Calculate the number of non zero values in the sparse matrix. This number is known from the structure of the sparse matrix
	nzv = 3 * dimX - 6;
	
	//@@ Insert the code to allocate the temp, tmp and the sparse matrix arrays using Unified Memory
	cputimer_start();
	cudaMallocManaged((void**)&temp, dimX * sizeof(double));
	cudaMallocManaged((void**)&A, nzv * sizeof(double));
	cudaMallocManaged((void**)&ARowPtr, (dimX + 1) * sizeof(int));
	cudaMallocManaged((void**)&AColIndx, nzv * sizeof(int));
	cudaMallocManaged((void**)&tmp, dimX * sizeof(double));
	cputimer_stop("Allocating device memory");

	// Check if concurrentAccessQ is non zero in order to prefetch memory
	if (concurrentAccessQ)
	{
		//@@ Insert code to prefetch in Unified Memory asynchronously to CPU
		cputimer_start();
		cudaMemPrefetchAsync(temp, dimX * sizeof(double), cudaCpuDeviceId); // TODO: should this be removed?
		cudaMemPrefetchAsync(A, nzv * sizeof(double), cudaCpuDeviceId);
		cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), cudaCpuDeviceId);
		cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), cudaCpuDeviceId);
		cudaMemPrefetchAsync(tmp, dimX * sizeof(double), cudaCpuDeviceId); // TODO: should this be removed?
		cputimer_stop("Prefetching GPU memory to the host");
	}

	// Initialize the sparse matrix
	cputimer_start();
	matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
	cputimer_stop("Initializing the sparse matrix on the host");

	//Initiliaze the boundary conditions for the heat equation
	cputimer_start();
	memset(temp, 0, sizeof(double) * dimX);
	temp[0] = tempLeft;
	temp[dimX - 1] = tempRight;
	cputimer_stop("Initializing memory on the host");

	if (concurrentAccessQ)
	{
		//@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
		cputimer_start();
		cudaMemPrefetchAsync(temp, dimX * sizeof(double), 0); // TODO: should this be removed?
		cudaMemPrefetchAsync(A, nzv * sizeof(double), 0);
		cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), 0);
		cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), 0);
		cudaMemPrefetchAsync(tmp, dimX * sizeof(double), 0); // TODO: should this be removed?
		cputimer_stop("Prefetching GPU memory to the device");
	}

	//@@ Insert code to create the cuBLAS handle
	cublasCreate_v2(&cublasHandle);

	//@@ Insert code to create the cuSPARSE handle
	cusparseCreate(&cusparseHandle);

	//@@ Insert code to set the cuBLAS pointer mode to CUSPARSE_POINTER_MODE_HOST
	//cublasSetPointerMode_v2(cublasHandle, (cublasPointerMode_t)CUSPARSE_POINTER_MODE_HOST);

	//@@ Insert code to call cusparse api to create the mat descriptor used by cuSPARSE
	//cusparseCreateMatDescr(&Adescriptor);

	//@@ Insert code to call cusparse api to get the buffer size needed by the sparse matrix per vector (SMPV) CSR routine of cuSPARSE
	//cusparseEstim

	//@@ Insert code to allocate the buffer needed by cuSPARSE

	// Perform the time step iterations
	for (int it = 0; it < nsteps; ++it) {
		//@@ Insert code to call cusparse api to compute the SMPV (sparse matrix multiplication) for
		//@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
		//@@ tmp = 1 * A * temp + 0 * tmp

		//@@ Insert code to call cublas api to compute the axpy routine using cuBLAS.
		//@@ This calculation corresponds to: temp = alpha * tmp + temp

		//@@ Insert code to call cublas api to compute the norm of the vector using cuBLAS
		//@@ This calculation corresponds to: ||temp||

		// If the norm of A*temp is smaller than 10^-4 exit the loop
		if (norm < 1e-4)
			break;
	}

	// Calculate the exact solution using thrust
	thrust::device_ptr<double> thrustPtr(tmp);
	thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
		(tempRight - tempLeft) / (dimX - 1));

	// Calculate the relative approximation error:
	one = -1;
	//@@ Insert the code to call cublas api to compute the difference between the exact solution
	//@@ and the approximation
	//@@ This calculation corresponds to: tmp = -temp + tmp

	//@@ Insert the code to call cublas api to compute the norm of the absolute error
	//@@ This calculation corresponds to: || tmp ||

	error = norm;
	//@@ Insert the code to call cublas api to compute the norm of temp
	//@@ This calculation corresponds to: || temp ||

	// Calculate the relative error
	error = error / norm;
	printf("The relative error of the approximation is %f\n", error);

	//@@ Insert the code to destroy the mat descriptor
	//cusparseDestroyMatDescr(Adescriptor);

	//@@ Insert the code to destroy the cuSPARSE handle
	cusparseDestroy(cusparseHandle);

	//@@ Insert the code to destroy the cuBLAS handle
	cublasDestroy_v2(cublasHandle);

	//@@ Insert the code for deallocating memory
	cudaFree(temp);
	cudaFree(A);
	cudaFree(ARowPtr);
	cudaFree(AColIndx);
	cudaFree(tmp);

	return 0;
}