// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>


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

void verify(float *A, float *B, float *C, int m, int k, int n) {

  const float relativeTolerance = 1e-1;

  for(int row = 0; row < m; ++row) {
    for(int col = 0; col < n; ++col) {
      float sum = 0;
      for(int i = 0; i < k; ++i) {
        sum += A[row*k + i]*B[i*n + col];
      }
      float relativeError = (sum - C[row*n + col])/sum;
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
        std::cout<<"TEST FAILED"<<std::endl;        
        std::cout<<"sum: "<<sum<<"," << "C["<<row<<"]"<<"["<<col<<"]"<<":"<< C[row*n + col]<<std::endl;
        exit(0);
      }
    }
  }
        std::cout<<"TEST PASSED"<<std::endl;

}

/*
void verify(float *A, float *B, float *C, int m, int k, int n) {

  const float relativeTolerance = 1e-1;

  for(int row = 0; row < m; ++row) {
    for(int col = 0; col < n; ++col) {
      float sum = 0;
      for(int i = 0; i < k; ++i) {
        sum += A[row*k + i]*B[i*n + col];
      }
      float relativeError = (sum - C[row*n + col])/sum;
      if (relativeError > relativeTolerance
        || relativeError < -relativeTolerance) {
        std::cout<<"TEST FAILED"<<std::endl;        
        std::cout<<"sum: "<<sum<<"," << "C["<<row<<"]"<<"["<<col<<"]"<<":"<< C[row*n + col]<<std::endl;
        exit(0);
      }
    }
  }
        std::cout<<"TEST PASSED"<<std::endl;

}

*/
void build_checksum(float *cs_rows, float *cs_cols, float *C_A ,float *A, int nr_rows_A, int nr_cols_A)//is not working correctly
{

      	for (int i = 0; i < nr_rows_A; ++i) {
	    for (int j = 0; j < nr_cols_A; ++j)
		    {

		   cs_rows[i] += A[i * nr_cols_A + j];
		    //C_A[i*nr_cols_A+j]=A[i * nr_cols_A + j];
		   // C_A[nr_rows_A*nr_cols_A+j]

		    }
	      }
/*for (int j=0; j<nr_cols_A;j++){
C_A[nr_rows_A*(nr_cols_A+1)+j] = cs_rows[j];
}*/
	for (int j = 0; j < nr_cols_A; ++j) {
        	for (int i = 0; i < nr_rows_A; ++i)
			{
          //      C_A[i * nr_cols_A + j] = A[i * nr_cols_A + j];
			    cs_cols[j] += A[i * nr_cols_A + j];
			    //C_A[i*nr_cols_A+j]=A[i * nr_cols_A + j];
		      //  C_A[nr_rows_A*nr_cols_A+j] += A[i * nr_cols_A + j];

			}

    	}
for (int i=0;i<nr_rows_A;i++){
    for (int j=0;j<nr_cols_A;j++)
    {

    C_A[i*(nr_cols_A+1)+j]= A[i * nr_cols_A + j];
    C_A[i*(nr_cols_A+1)+j+1]= cs_rows[i];

    }
    }
for (int j=0;j<nr_cols_A;j++){
C_A[nr_rows_A*(nr_cols_A+1)+j] = cs_cols[j];
}

}


/* return value :
 * -2 : failure, too many faults
 * -1 : failure, unable to recover
 *  0 : pass
 *  1 : failure, recovered
 * */


/* return value :
 * -2 : failure, too many faults
 * -1 : failure, unable to recover
 *  0 : pass
 *  1 : failure, recovered
 * */

void matrix_input_generator(float *h_a, int m, int n){
// random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 4;
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
        for (unsigned int i=0; i < (nr_rows_C+1) * (nr_cols_C+1); i++) { h_C[i] = 0; }

	float *A_row_checksum= (float *)malloc(nr_cols_A * 1 * sizeof(float));
	    for (unsigned int i=0; i <  nr_cols_C; i++) { A_row_checksum[i] = 0; }

	float *A_col_checksum= (float *)malloc(1 * nr_rows_A * sizeof(float));
	    for (unsigned int i=0; i < nr_rows_C ; i++) { A_col_checksum[i] = 0; }	

	float *h_A_checksum= (float *)malloc((nr_rows_A+1) * (nr_cols_A+1) * sizeof(float));
	    for (unsigned int i=0; i < nr_rows_C * nr_cols_C; i++) { h_A_checksum[i] = 0; }

	float *h_B_checksum= (float *)malloc((nr_rows_B+1) * (nr_cols_B+1) * sizeof(float));
	    for (unsigned int i=0; i < (nr_rows_C+1) * (nr_cols_C+1); i++) { h_B_checksum[i] = 0; }

	float *h_C_checksum= (float *)malloc((nr_rows_C+1) * (nr_cols_C+1) * sizeof(float));
	    for (unsigned int i=0; i < (nr_rows_C+1) * (nr_cols_C+1); i++) { h_C_checksum[i] = 0; }

        float *B_row_checksum= (float *)malloc(nr_cols_B * 1 * sizeof(float));
	    for (unsigned int i=0; i <  nr_cols_B; i++) { B_row_checksum[i] = 0; }

	float *B_col_checksum= (float *)malloc(nr_rows_B * sizeof(float));
	    for (unsigned int i=0; i < nr_cols_B; i++) { B_col_checksum[i] = 0; }

	float *C_row_checksum= (float *)malloc(nr_cols_C *  sizeof(float));
	    for (unsigned int i=0; i < nr_cols_C; i++) { C_row_checksum[i] = 0; }

	float *C_col_checksum= (float *)malloc(nr_rows_C * sizeof(float));
	    for (unsigned int i=0; i < nr_rows_C ; i++) { C_col_checksum[i] = 0; }
	
	//float *h_AA = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	//float *h_BB = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	float *d_c_A, *d_c_B, *d_c_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

	cudaMalloc(&d_c_A, (nr_rows_A + 1) * (nr_cols_A + 1) * sizeof(float));
	cudaMalloc(&d_c_B, (nr_rows_B + 1) * (nr_cols_B + 1) * sizeof(float));
	cudaMalloc(&d_c_C, (nr_rows_C + 1) * (nr_cols_C + 1) * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	
	// Fill the arrays A and B on GPU with random numbers
	
	//GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);

	//GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	matrix_input_generator(h_A, nr_rows_A, nr_cols_A);
	matrix_input_generator(h_B, nr_rows_B, nr_cols_B);


	build_checksum(A_row_checksum, A_col_checksum, h_A_checksum, h_A, nr_rows_A, nr_cols_A);//just for test
	build_checksum(B_row_checksum, B_col_checksum, h_B_checksum, h_B, nr_rows_B, nr_cols_B);//just for test
	
	 cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);


	std::cout << "A in cpu side=" << std::endl;
	print_matrix(h_A, nr_rows_A, nr_cols_A);
	
	std::cout << "Row checksum of Matrix A=" << std::endl;//for test
	print_matrix(A_row_checksum, nr_rows_A, 1);//for test

	std::cout << "Col checksum of Matrix A=" << std::endl;//for test	
	print_matrix(A_col_checksum, 1, nr_cols_A);//for test


	std::cout << "B in cpu side=" << std::endl;
	print_matrix(h_B, nr_rows_B, nr_cols_B);

	std::cout << "Row checksum of Matrix B=" << std::endl;//for test
	print_matrix(B_row_checksum, nr_rows_B, 1);//for test

	std::cout << "Col checksum of Matrix B=" << std::endl;//for test	
	print_matrix(B_col_checksum, 1, nr_cols_B);//for test

	
	 cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);//added for moving random values to GPU side
	 cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);//added for moving random values to GPU side

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
	//std::cout << "A after gpu->cpu=" << std::endl;
	//print_matrix(h_A, nr_rows_A, nr_cols_A);
	//std::cout << "B after gpu->cpu=" << std::endl;
	//print_matrix(h_B, nr_rows_B, nr_cols_B);

	// Multiply A and B on GPU
	cudaEventRecord(start);	
	gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	float *cb_result = (float*) malloc (sizeof(float));
	float *cb_result2= (float *) malloc( sizeof(float));
	cudaEventRecord(start);	
	gpu_checksum(cb_result, h_C_checksum, (nr_rows_C) * (nr_cols_C));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	std::cout << "the result matrix sum is =" << cb_result<<std::endl;

	//fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds);

	// Copy (and print) the result on host memory
	cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
	std::cout << "C =" << std::endl;
	print_matrix(h_C, nr_rows_C, nr_cols_C);

//matrix multiplication with checksum
/*
	std::cout << "Matrix A with checksum =" << std::endl;//for test	
	print_matrix(h_A_checksum, nr_rows_A+1, nr_cols_A+1);//for test
	
	std::cout << "Matrix B with checksum =" << std::endl;//for test	
	print_matrix(h_B_checksum, nr_cols_B+1, nr_cols_B+1);//for test

 	cudaMemcpy(d_c_A,h_A_checksum, (nr_rows_A + 1) * (nr_cols_A + 1) * sizeof(float),cudaMemcpyHostToDevice);//added for moving random values to GPU side
	cudaMemcpy(d_c_B,h_B_checksum, (nr_rows_B + 1) * (nr_cols_B + 1) * sizeof(float),cudaMemcpyHostToDevice);//added for moving random values to GPU side

cudaEventRecord(start);	
	gpu_blas_mmul(d_c_A, d_c_B, d_c_C, (nr_rows_A+1), nr_cols_A+1, (nr_cols_B+1));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds2 = 0;
	cudaEventElapsedTime(&milliseconds2, start, stop);
	
	//fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds);
	
	// Copy (and print) the result on host memory
	cudaMemcpy(h_C_checksum,d_c_C,(nr_rows_C+1) * (nr_cols_C+1) * sizeof(float),cudaMemcpyDeviceToHost);
	std::cout << "C with checksum =" << std::endl;
	print_matrix(h_C_checksum, nr_rows_C+1, nr_cols_C+1);
*/


	//verify(h_A, h_B, h_C, nr_rows_A, nr_cols_A, nr_cols_B); //test correctness of code
	
	// added_start
	//cudaEventRecord(start);	
	//gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds2 = 0;
	//cudaEventElapsedTime(&milliseconds2, start, stop);
	// std::cout << "MMUL_1 Execution completed. Elapsed Time = " << milliseconds2 << std::endl;
	//fprintf(stdout, "Execution completed. Elapsed Time = %6.8f msecs\n", milliseconds2);

	// Copy (and print) the result on host memory
	//cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
	//std::cout << "C2 =" << std::endl;
	//print_matrix(h_C, nr_rows_C, nr_cols_C);
	
	// added_end
	 std::cout << "MMUL_1 Execution completed. Elapsed Time = " << milliseconds << std::endl;
	//std::cout << "MMUL_1 Done" << std::endl;
	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);	
	cudaFree(d_c_A);
	cudaFree(d_c_B);
	cudaFree(d_c_C);	
	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(A_row_checksum);
	free(A_col_checksum);
 	free(B_row_checksum);
	free(B_col_checksum);
	free(C_row_checksum);
	free(C_col_checksum);

	return 0;
}
