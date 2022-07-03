#include <stdio.h>
#include <cuda.h>
__host__ __device__ void dhfun() {
	printf("I can run on both CPU and GPU.\n");
}
__device__ unsigned dfun(unsigned *vector, unsigned vectorsize, unsigned id) {
	if (id == 0) dhfun();
	if (id < vectorsize) {
		vector[id] = id;
		return 1;
	} else {
		return 0;
	}
}
__global__ void dkernel(unsigned *vector, unsigned vectorsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dfun(vector, vectorsize, id);
}
__host__ void hostfun() {
	printf("I am simply like another function running on CPU. Calling dhfun\n");
	dhfun();
}
#define BLOCKSIZE	1024
int main(int nn, char *str[]) {
	unsigned N = atoi(str[1]);
	unsigned *vector, *hvector;
	cudaMalloc(&vector, N * sizeof(unsigned));
	hvector = (unsigned *)malloc(N * sizeof(unsigned));

	unsigned nblocks = ceil((float)N / BLOCKSIZE);
	printf("nblocks = %d\n", nblocks);

    	dkernel<<<nblocks, BLOCKSIZE>>>(vector, N);
	cudaMemcpy(hvector, vector, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
	for (unsigned ii = 0; ii < N; ++ii) {
		printf("%4d ", hvector[ii]);
	}
	printf("\n");
	hostfun();
	dhfun();
    	return 0;
}
