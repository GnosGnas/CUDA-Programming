#include <stdio.h>
#include <cuda.h>

class A {
public:
	__host__ __device__ A(unsigned ongpu = 1) { printf("in A's constructor: on %s.\n", (ongpu ? "GPU" : "CPU")); }
};
__global__ void dkernel(unsigned n) {
	A a;
	//printf("in dkernel %d\n", blockIdx.x * blockDim.x + threadIdx.x);
}

#define BLOCKSIZE	32
int main() {
	A b(0);
	unsigned N = BLOCKSIZE;
    	dkernel<<<1, BLOCKSIZE>>>(N);
	cudaThreadSynchronize();
    	return 0;
}
