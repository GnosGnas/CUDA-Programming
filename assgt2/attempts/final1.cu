#include<iostream>
#include<sys/time.h>
#include<cuda.h>

#include <stdlib.h>

using namespace std;


// write kernels here...
// to add Atrans+B
//32x32 blocks used
__global__ void addmats(int *A, int *B, int *sumres, int rowmax, int colmax) {
	__shared__ int temp[32][32];
	
	//rowmax and colmax for result array
	//reading to cache and storing it as Atrans
	int rowidA = (blockIdx.x*32) + threadIdx.y;
	int colidA = (blockIdx.y*32) + threadIdx.x;
	int tidA = (rowidA*rowmax) + colidA;
	
	if ((rowidA<colmax) && (colidA<rowmax)) {	
		temp[threadIdx.x][threadIdx.y] = A[tidA]; //coalesced but bank conflict
	}
	__syncthreads();
	
	//adding
	int rowid = (blockIdx.y*32) + threadIdx.y;
	int colid = (blockIdx.x*32) + threadIdx.x;
	int tid = (rowid*colmax) + colid;
	
	if ((rowid<rowmax) && (colid<colmax))	{
		sumres[tid] = temp[threadIdx.y][threadIdx.x] + B[tid]; //coalesced and no conflicts
	}
}

//For A * Btrans
//16x16 blocks used
// A = aXb and B = cXb
__global__ void badmult(int *A, int *B, int *multres, int a, int b, int c) {
	__shared__ int temp[16][32];
	
	int rowidA = (blockIdx.y*16) + threadIdx.y;
	int rowidB = (blockIdx.x*16) + threadIdx.y;
	
	for(int k=0; k<b; k+=16) {
		//reading both to cache
		int colidA = k + threadIdx.x;
		int tidA = (rowidA*b) + colidA;
		
		if ((rowidA<a) && (colidA<b)) { //put it out
			temp[threadIdx.x][threadIdx.y] = A[tidA]; //bank conflicts but coalesced
		}
		
		int colidB = k + threadIdx.x;
		int tidB = (rowidB*b) + colidB;
		
		if ((rowidB<c) && (colidB<b)) {
			temp[threadIdx.x][threadIdx.y + 16] = B[tidB]; //bank conflicts but coalesced
		}
		__syncthreads();
		
		//multiplying
		int rowid = (blockIdx.y*16) + threadIdx.y;
		int colid = (blockIdx.x*16) + threadIdx.x;
		int tid = (rowid*c) + colid;
		
		for (int i=0; i<16 ;i++) {

			if ((blockIdx.y*16 + threadIdx.y < a)  && (k + i < b) && (blockIdx.x*16 + threadIdx.x < c)) {
				multres[tid] += temp[i][threadIdx.y] * temp[i][threadIdx.x + 16]; //No bank conflicts and coalesced - Also data from same bank accessed sequentially
			}
		}
		__syncthreads();
	}
}

//For Atrans*B
//32x32 blocks
//A = aXb  B = aXc
__global__ void goodmult(int *A, int *B, int *multres, int a, int b, int c) {	
	int imm[32][32];
	int immtemp;
	int colidA = (blockIdx.y*32) + threadIdx.y;
	int colidB = (blockIdx.x*32) + threadIdx.x;
	
	imm[threadIdx.y][threadIdx.x] = 0;
	
	if ((colidA<b) && (colidB<c)) {
		for(int blockiter=0; blockiter<2+(b/32); blockiter++) {
			immtemp = 0;
			
			for(int i=0; i<32; i++) {
				int rowidA = (blockiter*32) + i;
				int tidA = (rowidA*b) + colidA;
				
				int rowidB = (blockiter*32) + i;
				int tidB = (rowidB*c) + colidB;
				
				if ((rowidA<a) && (rowidB<a)) {
					immtemp += A[tidA]*B[tidB];
				}
			}
			imm[threadIdx.y][threadIdx.x] += immtemp;
		}
		
		int rowid = (blockIdx.y*32) + threadIdx.y;
		int colid = (blockIdx.x*32) + threadIdx.x;
		int tid = (rowid*c) + colid;
		
		multres[tid] = imm[threadIdx.y][threadIdx.x];
	}
}


// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	// variable declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX;
	int *addres, *multres;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * p * sizeof(int));
	cudaMalloc(&d_matrixC, q * r * sizeof(int));
	cudaMalloc(&d_matrixD, s * r * sizeof(int));
	cudaMalloc(&d_matrixX, p * s * sizeof(int));
	
	cudaMalloc(&addres, q * p * sizeof(int));
	cudaMalloc(&multres, q * s * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);
	
	// call the kernels for doing required computations...
	//1. Computing Atrans+B
	dim3 add_mats(2+(p/32), 2+(q/32), 1);
	addmats<<<add_mats, dim3(32, 32, 1)>>>(d_matrixA, d_matrixB, addres, q, p);

	//2. Computing C*Dtrans
	dim3 badmult_dims(2+(s/16), 2+(q/16), 1);
	badmult<<<badmult_dims, dim3(16, 16, 1)>>>(d_matrixC, d_matrixD, multres, q, r, s);
	cudaDeviceSynchronize(); //problematic as it syncs even addmats
	
	//3. Computing (add_res)trans * multres1
	dim3 goodmult_dims(2+(s/32), 2+(p/32), 1);
	goodmult<<<goodmult_dims, dim3(32, 32, 1)>>>(addres, multres, d_matrixX, q, p, s);
	cudaDeviceSynchronize();

	// copy the result back...
	cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
	
	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixX);
	
	cudaFree(addres);
	cudaFree(multres);
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
