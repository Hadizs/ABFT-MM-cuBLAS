// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(unsigned int *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	//curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);//? should be changed to integer values
	
	curandGeneratePoisson( prng, A, nr_rows_A * nr_cols_A , 2);

//Read more at: http://docs.nvidia.com/cuda/curand/index.html#ixzz55blQafmN 
//Follow us: @GPUComputing on Twitter | NVIDIA on Facebook

}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const float m, const float k, const float n) {
	int lda=m,ldb=k,ldc=m;
	const unsigned int alf = 1;
	const unsigned int bet = 0;
	const unsigned int *alpha = &alf;
	const unsigned int *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const int *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv) {
	
	srand(time(NULL));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate 3 arrays on CPU

	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
	nr_rows_A= atoi(argv[1]);
	nr_cols_A=nr_cols_B=nr_rows_B=nr_rows_C=nr_rows_B=nr_cols_C=nr_rows_A;
	
	// for simplicity we are going to use square arrays
	//nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 10000;
	
	int *h_A = (int *)malloc(nr_rows_A * nr_cols_A * sizeof(int));
	int *h_B = (int *)malloc(nr_rows_B * nr_cols_B * sizeof(int));
	int *h_C = (int *)malloc(nr_rows_C * nr_cols_C * sizeof(int));

	// Allocate 3 arrays on GPU
	unsigned int  *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(int));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(int));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(int));

	// If you already have useful values in A and B you can copy them in GPU:
	// cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	// cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(int),cudaMemcpyDeviceToHost);
	//std::cout << "A =" << std::endl;
	//print_matrix(h_A, nr_rows_A, nr_cols_A);
	//std::cout << "B =" << std::endl;
	//print_matrix(h_B, nr_rows_B, nr_cols_B);

	// Multiply A and B on GPU
	cudaEventRecord(start);	
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	 std::cout << "MMUL_1 Execution completed. Elapsed Time = " << milliseconds << std::endl;
	//fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(int),cudaMemcpyDeviceToHost);
	std::cout << "C =" << std::endl;
	print_matrix(h_C, nr_rows_C, nr_cols_C);
	
	// added_start
	cudaEventRecord(start);	
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds2 = 0;
	cudaEventElapsedTime(&milliseconds2, start, stop);
	 std::cout << "MMUL_1 Execution completed. Elapsed Time = " << milliseconds2 << std::endl;
	//fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds2);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(int),cudaMemcpyDeviceToHost);
	std::cout << "C2 =" << std::endl;
	print_matrix(h_C, nr_rows_C, nr_cols_C);
	
	// added_end

	//std::cout << "MMUL_1 Done" << std::endl;
	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
