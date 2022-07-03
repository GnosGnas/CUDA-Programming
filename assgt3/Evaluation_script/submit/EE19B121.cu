#include <stdio.h>
#include <cuda.h>

using namespace std;

//Complete the following function
__global__ void gpu_operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	__shared__ int exit_flag;
	__shared__ int curr_task; //prolly need not be shared
	__shared__ unsigned int task_ctr;
	__shared__ int corenum; //used to find min free core
	__shared__ int last_core;
	__shared__ int cut_off;
	__shared__ int last_tid;
	
	
	__shared__ int tasktocore[1024];
	__shared__ int ends[1024];
	__shared__ int ptocore[1024]; //priority to core mapping
	
	int tid = threadIdx.x;
	
	int local_flag = 0; 	// if a task is scheduled for the thread it will be set as 1
	int exetask; 			// task number which is to be executed by the thread
	int ref_tid;			// Just a relative position among the tasks getting scheduled in parallel
	int core_free = 1;		// Flag to tell if a core went into wait mode
	int go_flag;			// temp variable so that I dont have to check if assigned_priority==priority[task] multiple times
	int prev;				// This is relative to the tasks scheduled in parallel
	int prev_tid;			// This is the tid which is executing the prev task
	int cut_off1 = 0;		// 
	int startingtime_newtask = 0;
	int delayer=0;


	task_ctr = 0;
	ends[tid] = 0;
	ptocore[tid] = m;
	last_tid = -1;
	
	while (task_ctr < n) {
		exit_flag = 0;
		ref_tid = -1;
		
		// Scheduling of tasks to cores - this is a sequential execution and only one thread executes at a time
		for (int i=0; (i<m+1) && (task_ctr < n); i++) {
			curr_task = task_ctr;
			corenum = m;
			go_flag = 0;
			__syncthreads();
			
			if (ptocore[priority[curr_task]] == m) {
				if (delayer==1) {
					if (core_free == 1) {
						atomicMin(&corenum, tid);
					}
					delayer = 0;
				}
				else {
					delayer = 1;
					exit_flag = 1;
					if (i!=0) {
						last_core = tasktocore[i-1];
					}
				}
			}
			else if (ptocore[priority[curr_task]] == tid) go_flag = 1;
			__syncthreads();
			
			if (corenum == tid) {
				ptocore[priority[curr_task]] = tid;
				corenum = m;
				go_flag = 1;
				ends[tid] = startingtime_newtask;
			}
			
			if (go_flag == 1) {
				if (local_flag==1) {
				 exit_flag = 1;
				 last_core = tid; 
				 last_tid = tasktocore[i-1];
				}
				else {
					core_free = 0;
					tasktocore[i] = tid;
					ref_tid = i;
					local_flag = 1;
					exetask = curr_task;
					task_ctr++;
				}
			}
		   __syncthreads();
		   
			if (exit_flag == 1)
				break;
		}
		__syncthreads();
		
		
		// Prefix computation for finding when a task should start executing - ends[tid] is the end of last task on core tid
		//2 constraints for a task to start - should start after the prev task in the queue starts and should start after the prev task in the core ends
		for (int i=1; i<m; i*=2) { 
			if (ref_tid >= i) { 
				int val = (ref_tid)%(2*i);
				if (val >= i) {

					if (local_flag==1) {
						prev = ref_tid - val + i-1;
						prev_tid = tasktocore[prev];
						if (ends[prev_tid] > ends[tid])
							ends[tid] = ends[prev_tid];
					}

				}
			}
			__syncthreads();
		}
		
		
		// Handling of special cases - mostly sequential
		if (delayer==1) {
			if (cut_off1 > ends[last_core])
				startingtime_newtask = cut_off1;
			else
				startingtime_newtask = ends[last_core];
		}
	
		if (last_tid==tid) {
			cut_off = ends[last_tid];
			last_tid = -1;
		}
		__syncthreads();
		
		// Updation of ends[tid] - parallel
		if (local_flag == 1) {
			if (cut_off1 > ends[tid])
				ends[tid] = cut_off1 + executionTime[exetask];
			else
				ends[tid] += executionTime[exetask];
			
			result[exetask] = ends[tid];
			local_flag = 0;
		}
		cut_off1 = cut_off;
		__syncthreads();
		
		
		// Freeing cores - parallel
		if (delayer==1) {
			if (startingtime_newtask >= ends[tid])
				core_free = 1;
		}
		else if ((ends[last_core] > ends[tid]) || ((ends[last_core] == ends[tid])&&(last_core!=tid))) {
			core_free = 1;
		}
	}	
}


void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	int * d_executionTime, *d_priority, *d_result;
	
	cudaMalloc(&d_executionTime, n * sizeof(int));
	cudaMalloc(&d_priority, n * sizeof(int));
	cudaMalloc(&d_result, n * sizeof(int));
	
	cudaMemcpy(d_executionTime, executionTime, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_priority, priority, n * sizeof(int), cudaMemcpyHostToDevice);	

	gpu_operations<<<1, m>>>(m, n, d_executionTime, d_priority, d_result);

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
