// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <sys/time.h>
#include <fstream>
using namespace std;


/* return value :
 * -2 : failure, too many faults
 * -1 : failure, unable to recover
 *  0 : pass
 *  1 : failure, recovered
 * */
void gpu_checksum(float *result , float *matrix, int matrixSize){

cublasStatus_t ret;  
cublasHandle_t handle;
ret = cublasCreate(&handle);

ret = cublasSasum(handle, matrixSize, matrix, 1, result);
 
cublasDestroy(handle);

}

bool verify(float a, float b) {
bool test_result=0;
  const float relativeTolerance = 1e-1;

      float relativeError = ((a - b)/a);
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
	test_result=0;
        std::cout<<"TEST FAILED"<<std::endl;        

        exit(0);
      }
	test_result=1;
        std::cout<<"TEST PASSED"<<std::endl;
	return test_result;
    
}
  


float build_checksum(float *cs_A_rows, float *A, float *cs_B_cols, float *B, int nr_rows_A, int nr_cols_A,int nr_rows_B,int nr_cols_B )//is not working correctly
{
float checksum_sum=0;
      	for (int i = 0; i < nr_rows_A; ++i) {
	    for (int j = 0; j < nr_cols_A; ++j)
		    {
		   cs_A_rows[i] += A[i * nr_cols_A + j];
		    }
	      }

	for (int j = 0; j < nr_cols_B; ++j) {
        	for (int i = 0; i < nr_rows_B; ++i)
			{
        	    cs_B_cols[j] += B[i * nr_cols_B + j];
			}
}
for (int i=0; i< nr_cols_A;i++ ){
	checksum_sum += cs_A_rows[i]*cs_B_cols[i];
}
return checksum_sum;


}



void matrix_input_generator(float *h_a, int m, int n){
// random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

}
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv) {
	
	  ofstream myfile;
	  myfile.open ("exe_time_abft_mm_cublas.txt",ios::app );
	  myfile << "Writing this to a file.\n";
	  myfile << "Writing this to a file 2.\n";
	
	
        struct timeval t1, t2;
        float elapsedTime;
	//FILE *fpo = fopen("exe_time_bfs.txt\n\n\n\n","a");

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
	
	float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	for (unsigned int i=0; i < nr_rows_A * nr_cols_A; i++) { h_A[i] = 0; }

	float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	for (unsigned int i=0; i < nr_rows_B * nr_cols_B; i++) { h_B[i] = 0; }

	float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
        for (unsigned int i=0; i < (nr_rows_C) * (nr_cols_C); i++) { h_C[i] = 0; }

	float *A_row_checksum= (float *)malloc(nr_cols_A * 1 * sizeof(float));
	    for (unsigned int i=0; i <  nr_cols_C; i++) { A_row_checksum[i] = 0; }



	float *B_col_checksum= (float *)malloc(nr_rows_B * sizeof(float));
	    for (unsigned int i=0; i < nr_cols_B; i++) { B_col_checksum[i] = 0; }

	
	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;

	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

	
	// If you already have useful values in A and B you can copy them in GPU:
	

	// Fill the arrays A and B on GPU with random numbers
	
	//GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	//GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	matrix_input_generator(h_A, nr_rows_A, nr_cols_A);
	matrix_input_generator(h_B, nr_rows_B, nr_cols_B);

	float matrix_checksum=0;

	
	 cudaMemcpy(d_A , h_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice);
	 cudaMemcpy(d_B , h_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice);

	gettimeofday(&t1, NULL);

	matrix_checksum = build_checksum(A_row_checksum, h_A, B_col_checksum, h_B, nr_rows_A, nr_cols_A,  nr_rows_B, nr_cols_B);	

        gettimeofday(&t2, NULL);
	

	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms


	
//	fclose(fpo);


/*
	std::cout << "A in cpu side=" << std::endl;
	print_matrix(h_A, nr_rows_A, nr_cols_A);
	
	std::cout << "Row checksum of Matrix A=" << std::endl;//for test
	print_matrix(A_row_checksum, nr_rows_A, 1);//for test

	std::cout << "B in cpu side=" << std::endl;
	print_matrix(h_B, nr_rows_B, nr_cols_B);

	std::cout << "Col checksum of Matrix B=" << std::endl;//for test	
	print_matrix(B_col_checksum, 1, nr_cols_B);//for test
*/
	std::cout << "A and B checksum summation=" << matrix_checksum << std::endl;
		
	
	// Multiply A and B on GPU
	cudaEventRecord(start);	
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	float cb_result =0;

	cudaEventRecord(start);	
	gpu_checksum(&cb_result, d_C, (nr_rows_C) * (nr_cols_C));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds2 = 0;
	cudaEventElapsedTime(&milliseconds2, start, stop);

	std::cout << "the result matrix sum in GPU is =" << cb_result<<std::endl;

	verify(matrix_checksum, cb_result);

        myfile << "Execution Time for Matrix M * N * K = "<<nr_rows_A<<" * "<<nr_cols_A<<" * "<<nr_cols_B<<" * "<<std::endl<<std::endl;
	myfile << "Multiplication Execution Time without checksum = " << milliseconds <<" "<< "ms"<<std::endl;
	myfile << "Checksum calculation in CPU                    = " << elapsedTime << " "<< "ms"<<std::endl;
	myfile << "Checksum Calculation Execution Time in GPU     = " << milliseconds2 <<" " << "ms"<<std::endl;
	myfile << "Total Kernel Execution Time with checksum      = " << milliseconds+milliseconds2 <<" "<< "ms"<< std::endl;

	std::cout << "Multiplication Execution Time without checksum = " << milliseconds <<" "<< "ms"<<std::endl;
	std::cout << "Checksum calculation in CPU                    = " << elapsedTime << " "<< "ms"<<std::endl;
	std::cout << "Checksum Calculation Execution Time in GPU     = " << milliseconds2 <<" " << "ms"<<std::endl;
	std::cout << "Total Kernel Execution Time with checksum      = " << milliseconds+milliseconds2 <<" "<< "ms"<< std::endl<<std::endl<<std::endl;

	// Copy (and print) the result on host memory
	
	cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
	//std::cout << "C =" << std::endl;
	//print_matrix(h_C, nr_rows_C, nr_cols_C);


	
	// added_end
//	 std::cout << "MMUL_1 Execution completed. Elapsed Time = " << milliseconds << std::endl;
	//std::cout << "MMUL_1 Done" << std::endl;
	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	
	//cudaFree(d_c_A);
	//cudaFree(d_c_B);
	//cudaFree(d_c_C);	
	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(A_row_checksum);
	//free(A_col_checksum);
 	//free(B_row_checksum);
	free(B_col_checksum);
	//free(C_row_checksum);
	//free(C_col_checksum);
	myfile.close();
	return 0;
}
