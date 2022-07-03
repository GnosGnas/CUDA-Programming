/* In this attempt:
1. normal per element mat mult done and separately with cds() in b/w
2. extra dtemp array declared for intermediate result
//ignore results
Time taken (ms): 494.364 
success
Time taken (ms): 434.195
success
*/

#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;


// write kernels here...
__global__ void abcmult(int *dA, int *dB, int *dC, int p, int q, int r, int *dtemp) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int rowid = tid/r;
	int colid = tid%r;
	
	if (tid < p*r) {
		for (int i=0; i<q; i++)
			dtemp[tid] += (dA[(rowid*q) + i] + dB[(i*p) + rowid]) * dC[(i*r) + colid];
	}
}

__global__ void dmult(int *dtemp, int *dD, int *dX, int p, int r, int s) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int rowid = tid/s;
	int colid = tid%s;
	
	if (tid < p*s) {
		for (int i=0; i<r; i++)
			dX[tid] += dtemp[(rowid*r) + i] * dD[(colid*r) + i];
	}
}

// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX;
	int *d_temp;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * p * sizeof(int));
	cudaMalloc(&d_matrixC, q * r * sizeof(int));
	cudaMalloc(&d_matrixD, s * r * sizeof(int));
	cudaMalloc(&d_matrixX, p * s * sizeof(int));
	
	cudaMalloc(&d_temp,    p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);
	
	// call the kernels for doing required computations...
	struct timeval t1, t2;
	double seconds, microSeconds;
	
	int blocksize = 1024;
	int gridsize1 = 1 + (p*r)/blocksize;
	int gridsize2 = 1 + (p*s)/blocksize;
	
	gettimeofday(&t1, NULL);
	abcmult<<<blocksize, gridsize1>>>(d_matrixA, d_matrixB, d_matrixC, p, q, r, d_temp);
	cudaDeviceSynchronize();
	dmult<<<blocksize, gridsize2>>>(d_temp, d_matrixD, d_matrixX, p, r, s);
	gettimeofday(&t2, NULL);
	
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken by mine: %.3f ms\n", 1000*seconds + microSeconds/1000);
	// copy the result back...
	cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
	
	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixX);
	cudaFree(d_temp);
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
