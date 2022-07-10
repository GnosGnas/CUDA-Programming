_global__ void per_row_column_kernel(long int *A, long int *B, long int *C,long int m, long int n){
	
    long int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadId < m){

      for(long int i = 0; i < n ;i++){

          C[threadId * n + i] = (A[threadId * n + i] + B[i * m + threadId]) * (B[i * m + threadId] - A[threadId * n +i]);

      }

    }
}

__global__ void per_column_row_kernel(long int *A, long int *B, long int *C,long int m, long int n){

    long int threadId = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y;

    if(threadId < n){

      for(long int i = 0; i < m ;i++){

          C[i * n + threadId] = (A[i * n + threadId] + B[threadId * m + i]) * (B[threadId * m + i] - A[i * n + threadId]);
      }

    }
}


__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){

    long int threadId =  (blockIdx.x * gridDim.y + blockIdx.y) * (blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y;

    long int row =  threadId / n;

    long int col =  threadId % n;

    if(row < m && col < n){

        C[row * n + col] = (A[row * n + col] + B[col * m + row]) * (B[col * m + row] - A[row * n + col]);
    }

}
