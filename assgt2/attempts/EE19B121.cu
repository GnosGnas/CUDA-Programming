#include<iostream>
#include<sys/time.h>
#include<cuda.h>

#include <stdlib.h>

using namespace std;

// #defines declaration
// The following are the block dimensions used for each kernel
// More details about each is given with the appropriate kernel. Maximum number of threads were tried to ensure per block to improve parallelism
#define ADD_MATS_Dim 32
#define A_Btrans_Dim 16
#define ATrans_B_Dim 32

/* Aim: (A+B.transpose)*C*D.transpose
Steps involved:
1. Allocate memory in the GPU and copy the values
2. Compute A.transpose+B and store in add_resut
2. Compute C*D.transpose and store in mult_result
3. Compute add_result.transpose*mult_result and store it in d_matrixX
4. Copy back the outputs into CPU memory
*/ 


// write kernels here...
/* 
Kernel#1 - add_stage(): Kernel function to compute A.transpose+B. The reason for computing this instead of A+B.transpose is because it is faster to compute A.transpose*B which is done in the third step.
Time taken for computing A.transpose+B and A+B.transpose is same so this speedens up greatly in the final multiplication. Each block of threads (32x32) operate on a tiled block of the matrices A and B and stores the result.
32x32 was taken because of the upper limit on number of threads in each of 1024.
Parameters:
1. Sum_result - Result is stored here
2. max_rows/max_cols - total number of rows/cols in the resulting matrix
*/
__global__ void add_stage(int *A, int *B, int *Sum_result, int max_rows, int max_cols) {
	__shared__ int temp[ADD_MATS_Dim][ADD_MATS_Dim]; // Shared memory to store tiles of MatrixA
	
	// MatrixA cannot be accessed in a coalesced manner unlike MatrixB. 
	// Hence we access MatrixA in a coalesced manner and store it in the shared memory and then access the shared memory to compute the result.
	// Reading MatrixA and storing it in shared memory
	int rowidA = (blockIdx.x*ADD_MATS_Dim) + threadIdx.y;
	int colidA = (blockIdx.y*ADD_MATS_Dim) + threadIdx.x;
	int tidA = (rowidA*max_rows) + colidA;
	
	if ((rowidA<max_cols) && (colidA<max_rows)) 
		temp[threadIdx.x][threadIdx.y] = A[tidA]; // Here the memory is accessed in a coalesced manner but there will be bank conflicts as we are storing the transpose in temp
	__syncthreads();
	
	
	// Computing and storing the result
	int rowid = (blockIdx.y*ADD_MATS_Dim) + threadIdx.y;
	int colid = (blockIdx.x*ADD_MATS_Dim) + threadIdx.x;
	int tid = (rowid*max_cols) + colid;
	
	if ((rowid<max_rows) && (colid<max_cols))
		Sum_result[tid] = temp[threadIdx.y][threadIdx.x] + B[tid]; // Here the MatrixB is accessedin a coalesced manner and there are no bank conflicts expected to happen
}


/*
Kernel#2 - a_mult_btranspose(): Kernel function to compute A*B.transpose. This function is used to compute C*D.transpose.
For this computation, each block of threads first reads matrices in a coalesced manner and stores them in the shared memory and then computes the result. Since two matrices were to be stored and there are only 32 shared memory banks, 16x16 tiles were used.
Parameters:
1. Mult_result - Result is stored here
2. a, b, c - MatrixA is of dimensions axb and MatrixB is of dimensions cxb
*/
__global__ void a_mult_btranspose(int *A, int *B, int *Mult_result, int a, int b, int c) {
	__shared__ int temp[A_Btrans_Dim][2*A_Btrans_Dim]; // Shared memory to store tiles of MatrixA and MatrixB
	
	int rowidA = (blockIdx.y*A_Btrans_Dim) + threadIdx.y;
	int rowidB = (blockIdx.x*A_Btrans_Dim) + threadIdx.y;

	int rowid = (blockIdx.y*A_Btrans_Dim) + threadIdx.y;
	int colid = (blockIdx.x*A_Btrans_Dim) + threadIdx.x;
	int tid = (rowid*c) + colid;

	int Mult_result_temp=0;
	
	for(int k=0; k<b; k+=A_Btrans_Dim) {
		// Reading and storing tiles of MatrixA and MatrixB in cache
		// To enable coalesced memory read, we store the tiles of both MatrixA and MatrixB in to cache before computations
		int colidAB = k + threadIdx.x;
		int tidA = (rowidA*b) + colidAB;
		int tidB = (rowidB*b) + colidAB;
		
		if (colidAB<b) {
			// Here we ensure coalesced access across a warp but there will be bank conflicts
			if (rowidA<a) temp[threadIdx.x][threadIdx.y] 				= A[tidA]; 
			if (rowidB<c) temp[threadIdx.x][threadIdx.y + A_Btrans_Dim] = B[tidB];
		}	
		__syncthreads();
		
		// Computing and storing the result
		// There are no bank conflicts in this and the data in each bank is accessed sequentially
		for (int i=0; i<A_Btrans_Dim ;i++)
			if ((blockIdx.y*A_Btrans_Dim + threadIdx.y < a)  && (k + i < b) && (blockIdx.x*A_Btrans_Dim + threadIdx.x < c))
				Mult_result_temp += temp[i][threadIdx.y] * temp[i][threadIdx.x + A_Btrans_Dim]; 
				
		__syncthreads();
	}
	
	// Writing back into memory in a coalesced manner
	if ((blockIdx.y*A_Btrans_Dim + threadIdx.y < a) && (blockIdx.x*A_Btrans_Dim + threadIdx.x < c)) 
		Mult_result[tid] = Mult_result_temp;
}


/*
Kernel#2 - atranspose_mult_b(): Kernel function to compute A.transpose*B. This function is used to compute the (A.transpose+B).transpose*(C*D.transpose).
Here we use tiled multiplication so that there is better utilization of cache and coalescing. Tiles of dimensions 32x32 were used as each block can have only 1024 threads.
Parameters:
1. Mult_result - Result is stored here
2. a, b, c - MatrixA is of dimensions axb and MatrixB is of dimensions axc
*/
__global__ void atranspose_mult_b(int *A, int *B, int *Mult_result, int a, int b, int c) {	
	int colidA = (blockIdx.y*ATrans_B_Dim) + threadIdx.y;
	int colidB = (blockIdx.x*ATrans_B_Dim) + threadIdx.x;
	int tid = (colidA*c) + colidB;
	
	int imm=0;
	
	if ((colidA<b) && (colidB<c)) {
		for(int blockiter=0; blockiter<1+(a/ATrans_B_Dim); blockiter++)
			for(int i=0; i<ATrans_B_Dim; i++) {
				int rowidA = (blockiter*ATrans_B_Dim) + i;
				int tidA = (rowidA*b) + colidA;
				
				int rowidB = (blockiter*ATrans_B_Dim) + i;
				int tidB = (rowidB*c) + colidB;
				
				if ((rowidA<a) && (rowidB<a))
					imm += A[tidA]*B[tidB];
			}

		Mult_result[tid] = imm;
	}
}


// function to compute the output matrix
// For computing time taken by execution of the kernels, uncomment the time variables and codes
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX;
	int *add_result, *mult_result;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * p * sizeof(int));
	cudaMalloc(&d_matrixC, q * r * sizeof(int));
	cudaMalloc(&d_matrixD, s * r * sizeof(int));
	cudaMalloc(&d_matrixX, p * s * sizeof(int));
	
	cudaMalloc(&add_result, q * p * sizeof(int));
	cudaMalloc(&mult_result, q * s * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);
	
	// call the kernels for doing required computations...
	//struct timeval t1, t2;
	//double seconds, microSeconds;
	
	//1. Computing Atrans+B
	//gettimeofday(&t1, NULL);
	dim3 add_stage_dims(1+(p/ADD_MATS_Dim), 1+(q/ADD_MATS_Dim), 1);
	add_stage<<<add_stage_dims, dim3(ADD_MATS_Dim, ADD_MATS_Dim, 1)>>>(d_matrixA, d_matrixB, add_result, q, p);
	
	//2. Computing C*Dtrans
	dim3 a_mult_btranspose_dims(1+(s/A_Btrans_Dim), 1+(q/A_Btrans_Dim), 1);
	a_mult_btranspose<<<a_mult_btranspose_dims, dim3(A_Btrans_Dim, A_Btrans_Dim, 1)>>>(d_matrixC, d_matrixD, mult_result, q, r, s);
	cudaDeviceSynchronize();
	
	//3. Computing (add_res)trans * multres1
	dim3 atranspose_mult_b_dims(1+(s/ATrans_B_Dim), 1+(p/ATrans_B_Dim), 1);
	atranspose_mult_b<<<atranspose_mult_b_dims, dim3(ATrans_B_Dim, ATrans_B_Dim, 1)>>>(add_result, mult_result, d_matrixX, q, p, s);
	//cudaDeviceSynchronize();
	//gettimeofday(&t2, NULL);

	//seconds = t2.tv_sec - t1.tv_sec;
	//microSeconds = t2.tv_usec - t1.tv_usec;
	//printf("Time taken by mine: %.3f ms\n", 1000*seconds + microSeconds/1000);

	// copy the result back...
	cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
	
	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixX);
	
	cudaFree(add_result);
	cudaFree(mult_result);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int));

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}
