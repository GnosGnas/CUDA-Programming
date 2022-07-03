#include <stdio.h>
#include <cuda.h>

using namespace std;

void operations1 ( int m, int n, int *executionTime, int *priority, int *result ){
    int core_ctr[m]; //keeps track of ends
    int core_pin[m]; //priority to core
    int core_cache = 0; //temp var - last blocking task

    for(int j = 0; j < m; j++){
        core_ctr[j] = 0; 
        core_pin[j] = -1; 
    }
    
    for(int i = 0; i < n; i++){
        if(core_pin[priority[i]] == -1){
            int min_val; 
            if(core_ctr[0] > core_cache)
                min_val = core_ctr[0];
            else
                min_val = core_cache; 
            int min_idx = 0; 
            int cmp_val = 0; 
            for(int j = 1; j <m; j++){
                if(core_ctr[j] > core_cache)
                    cmp_val = core_ctr[j]; 
                else
                    cmp_val = core_cache; 
                if(cmp_val < min_val){
                    min_val = cmp_val; 
                    min_idx = j; 
                }
            core_pin[priority[i]] = min_idx; 
            
            }
            printf("lol %d to %d\n", priority[i], min_idx);
        }

        int core_id = core_pin[priority[i]]; 

        if(core_ctr[core_id] > core_cache) {
            core_cache = core_ctr[core_id];
            printf("------------------------------\n");
           }
		printf("%d is doing %d (%d) exe\n", core_id, i, executionTime[i]);
        core_ctr[core_id] = core_cache + executionTime[i]; 
        result[i] = core_ctr[core_id];
        printf("res[%d] %d\n", i, result[i]);
    }
}

//Complete the following function
__global__ void gpu_operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	__shared__ int exit_flag;
	__shared__ int curr_task; //prolly need not be shared
	__shared__ unsigned int task_ctr;
	__shared__ int corenum; //used to find min free core
	__shared__ int last_core;
	__shared__ int cut_off;
	int cut_off1 = 0;
	
	__shared__ int tasktocore[1024];
	__shared__ int ends[1024];
	__shared__ int ptocore[1024]; //priority to core mapping
	
	int local_flag = 0; //if a task is scheduled for the thread it will be set as 1
	int exetask; //task number which is to be executed by the thread
	int tid = threadIdx.x;
	int ref_tid;  //Just a relative position among the tasks getting scheduled in parallel
	int core_free = 1;  //Flag to tell if a core went into wait mode
	int go_flag;  //temp variable so that I dont have to check if assigned_priority==priority[task] multiple times
	int prev; //This is relative to the tasks scheduled in parallel
	int prev_tid;  //This is the tid which is executing the prev task
	
	int temp_end = 0;
	
	int shit=0;
	int temp=0;


	task_ctr = 0;
	ends[tid] = 0;
	ptocore[tid] = m;
	
	while (task_ctr < n) {
		exit_flag = 0;
		ref_tid = -1;

		if (tid==0) printf("------------------------------\n");
		for (int i=0; (i<m+1) && (task_ctr < n); i++) {
			curr_task = task_ctr;
			corenum = m;
			go_flag = 0;
			__syncthreads();
			
			if (ptocore[priority[curr_task]] == m) {
				if (shit==1) {
					if (core_free == 1) {
						atomicMin(&corenum, tid);
					}
					shit = 0;
				}
				else {
					//if (tid==0) printf("Here\n");
					//i=0;
					shit = 1;
					exit_flag = 1;
					if (i!=0) {
						last_core = tasktocore[i-1];
					}
					else {
						temp_end = 0;
					} 
				}
			}
			else if (ptocore[priority[curr_task]] == tid) go_flag = 1;
			__syncthreads();
			
			if (corenum == tid) {
				ptocore[priority[curr_task]] = tid;
				corenum = m;
				go_flag = 1;
				ends[tid] = temp_end;
			}
			
			if (go_flag == 1) {
				if (local_flag==1) {
				 exit_flag = 1;
				 last_core = tid;  //This is just a reference needed to tell if other cores are free or not later
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


		//Prefix computation for finding when a task should start executing
		//2 constraints for a task to start - should start after the prev task in the queue starts and should start after the prev task in the core ends
		for (int i=1; i<m; i*=2) { 
			if (ref_tid >= i) { 
				int val = (ref_tid)%(2*i);
				if (val >= i) {

					if (local_flag==1) {
						prev = ref_tid - val + i-1;
						prev_tid = tasktocore[prev];

						if (ends[prev_tid] > ends[tid]) {
							ends[tid] = ends[prev_tid];

						}
					}

				}
			}
			__syncthreads();
		}
		
		if ( (shit==1)) {
			temp_end = ends[last_core];

		}
	
		
		if (local_flag == 1) {
			ends[tid] += executionTime[exetask];
			result[exetask] = ends[tid];

			local_flag = 0;
		}

		__syncthreads();
		
		if (shit==1) {
			if (temp_end >= ends[tid]) {
				core_free = 1;
			}
		}
		else 
		if ((ends[last_core] > ends[tid]) || ((ends[last_core] == ends[tid])&&(last_core!=tid))) { 
			core_free = 1;
		}
	}	
	__syncthreads();
}


void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	int * d_executionTime, *d_priority, *d_result;
	
	//
	int *myres = (int *) malloc ( (n) * sizeof (int) );
	operations1(m, n, executionTime, priority, myres);
	
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
	
	for (int i=0; i<n; i++) {
		if (myres[i] != result[i]) {
			printf("FAIL: at %d, (%d!=%d)\n", i, myres[i], result[i]);
			break;
		}
		
		if (i==n-1) printf("PASS");
	}
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
