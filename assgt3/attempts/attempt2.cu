#include <stdio.h>
#include <cuda.h>

using namespace std;



//Complete the following function
__global__ void gpu_operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	__shared__ int exit_flag;
	__shared__ int curr_task;
	__shared__ unsigned int task_ctr;
	
	__shared__ int coores2[1024];
	__shared__ int ends[1024];
	
	int local_flag = 0;
	int exetask;
	int tid = threadIdx.x;
	int ref_tid;

	task_ctr = 0;
	ends[tid] = 0;
	
	while (task_ctr < n) {
		exit_flag = 0;
		ref_tid = -1;
		__syncthreads();
		//if (tid==0) printf("------------------------------\n");
		for (int i=0; (i<m+1) && (task_ctr < n); i++) {
			curr_task = task_ctr;
		   __syncthreads();
			//printf("curr - %d\n", curr_task);
			if (tid == priority[curr_task]) {
				if (local_flag==1) {
				 exit_flag = 1;
				 //printf("localled twice-%d, %d, %d\n", tid, i, task_ctr);
				}
				else {
					coores2[i] = tid;
					ref_tid = i;
					local_flag = 1;
					exetask = curr_task;
					task_ctr++;
					//printf("%d, %d, exe\n", tid, task_ctr-1);
				}
			}
		   __syncthreads();
		   
			if (exit_flag == 1)
				break;
		}
		__syncthreads();
		//if (tid==0) printf("------------------------------\n");
		
		for (int i=1; i<m; i*=2) { 
			if (ref_tid >= i) { 
				int val = (ref_tid)%(2*i);
				if (val >= i) {
					if (local_flag==1) {
						int prev = ref_tid - val + i-1;
						int prev_tid = coores2[prev];
						//printf("now %d and (%d, %d) prev %d\n", tid, prev, i, prev_tid);
						if (ends[prev_tid] > ends[tid]) {
							ends[tid] = ends[prev_tid];
							//printf("prefix %d - %d\n", tid, ends[prev_tid]);
						}
					}
				}
			}
			__syncthreads();
		}
		
		if (local_flag == 1) {
			ends[tid] += executionTime[exetask];
			result[exetask] = ends[tid];
			//printf("%d - res[%d] %d\n", tid, exetask, result[exetask]);
			local_flag = 0;
		}
		__syncthreads();
	}	
}


void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	int * d_executionTime, *d_priority, *d_result;
	
	cudaMalloc(&d_executionTime, n * sizeof(int));
	cudaMalloc(&d_priority, n * sizeof(int));
	cudaMalloc(&d_result, n * sizeof(int));
	
	cudaMemcpy(d_executionTime, executionTime, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_priority, priority, n * sizeof(int), cudaMemcpyHostToDevice);	
	
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	float milliseconds = 0;
    cudaEventRecord(start,0);

	gpu_operations<<<1, m>>>(m, n, d_executionTime, d_priority, d_result);
	
	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken orignal: %.6f ms\n", milliseconds);
    
	cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_executionTime);
	cudaFree(d_priority);
	cudaFree(d_result);
}


int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}			
