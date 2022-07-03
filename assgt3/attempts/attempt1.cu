#include <stdio.h>
#include <cuda.h>

using namespace std;



//Complete the following function
__global__ void gpu_operations ( int m, int n, int *executionTime, int *priority, int *result )  {
	__shared__ int last_task;
	__shared__ int exit_flag;
	__shared__ int curr_task;
	__shared__ unsigned int task_ctr;
	
	int tempf = -1;
	int temp=-1;
	int local_flag = 0;
	int exetask;
	int tid = threadIdx.x;
	int exiter = 0;
	
	last_task = -1;
	task_ctr = 0;
	
	while (task_ctr < n) {
		curr_task = task_ctr;
		exit_flag = 0;
		exiter = 0;
		__syncthreads();
		if (tid==0) printf("------------------------------\n");
		for (int i=0; i<m+1; i++) {
			curr_task = task_ctr;
			printf("curr - %d\n", curr_task);
			__syncthreads();
			if (tid == priority[curr_task]) {
				if (local_flag==1) {
				 last_task = task_ctr-1;
				 exit_flag = 1;
				 exiter = 1;
				 printf("localled twice-%d, %d, %d\n", tid, i, task_ctr);
				}
				else {
					local_flag = 1;
					exetask = task_ctr;
					atomicInc(&task_ctr, n+10);
					//curr_task++;
					printf("%d, %d, exe\n", tid, task_ctr);
				}
			}
		   __syncthreads();
		   
			if (exit_flag == 1) {
				//printf("Working exit.. %d\n", i);
				break;
		   }
		}
		__syncthreads();
		//if (tid==0) printf("------------------------------\n");
		
		
		if (local_flag==1) {
			printf("%d - %d exe[%d] = %d\n", tid, temp, exetask, executionTime[exetask]);
			result[exetask] = executionTime[exetask];
			//local_flag = 0;
		   
			if (tempf==0) {
				printf("%d - for %d, res[%d] = %d\n", tid, exetask, temp, result[temp]);
				result[exetask] += result[temp];
			}
			printf("%d - res[%d] = %d\n", tid, exetask, result[exetask]);
			//temp = exetask;
		}
		if ((local_flag==1) && (exiter==1)) {temp = exetask; local_flag = 0;}
		else if ((local_flag==1) && (executionTime[exetask] > executionTime[last_task])) temp = exetask;
		else temp = last_task;
		printf("%d temp %d\n", tid,temp);
		//else
			//temp = last_task;
		tempf=0;
	}


	/*
	int last_task = -1;
	int coreid = -1;
	int flag=0;
	
	task_ctr = 0;
	__syncthreads();
	
	while (task_ctr < n) {
		if (tid == priority[task_ctr]) {
			flag = 1;
			atomicInc(task_ctr);
		}
		__syncthreads();
		
		if (tid == priority[task_ctr-1])) {
			result[task_ctr-1] = executionTime[task_ctr-1];
			
			if (last_task != -1) result[task_ctr-1] += result[last_task];
			
		}
		
		if (tid == priority[
	
	
	while (task_ctr < n) {
		flag=0;
		core_num = m;
		sync()
		
		if (coreid == priority[task_ctr])
			flag = 1;
		sync()
		
		if (flag == 0) {
			if (coreid = -1) {
				atomicMin(core_num, tid);
			}
		}
		sync()
		
		if (core_num == tid) {
			 if (flag == 0) {
				coreid = priority[task_ctr];
		
				result[task_ctr] = executionTime[task_ctr];
				if (last_task!=-1)	result[task_ctr] += result[last_task];
			}
		}
		
			
			
			
			
			
		if(coreid = -1)	
			core_num = tid;
		sync()
		
		if(coreid = -1)
			atomicMin(core_num, tid);
		sync()
				
		
		if (core_num == tid) {
			if(coreid == -1) {
				coreid = priority[task_ctr];
				result[task_ctr] = exectime[task_ctr];
				last_task = task_ctr;
				atomicInc(task_ctr);
			}
		}
	}
	
	
			
	*/
	
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
