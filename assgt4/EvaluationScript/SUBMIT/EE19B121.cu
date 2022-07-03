#include <stdio.h>
#include <cuda.h>
#include <time.h>

using namespace std;

__global__ void batch_process(
    int R, int *req_src, int *req_dst, int *req_tkt, int *req_trn, int *req_cls, int *track,
    int *dst, int *src, int *cap, unsigned int *req_stat, unsigned int *count, int *thread){

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; 
    
    int fail;  
    int cls;
    int trn;

    for(int i = 0; i < R; i++){
        fail = 0; 
        trn = req_trn[i];
        cls = req_cls[i];
		
        if(thread[i] == id){
            if(dst[trn] > src[trn]){
            	int j, seat_lt;
                for(j = req_src[i]; j < req_dst[i]; j++){
                    seat_lt = cap[25 * trn + cls]; 
                    if(track[1250 * trn + 50 * cls + j - src[trn]] + req_tkt[i] > seat_lt){
                        fail = 1; 
                        req_stat[i] = 0; 
                        atomicInc(&count[0], 5000); 
                        break; 
                    }
                }
                if(!fail){
                    for(int j = req_src[i]; j < req_dst[i]; j++){
                        track[1250 * trn + 50 * cls + j - src[trn]] += req_tkt[i];
                    } 
                    req_stat[i] = 1; 
                    atomicInc(&count[1], 5000); 
                    atomicAdd(&count[2], req_tkt[i] * (req_dst[i] - req_src[i])); 
                }
            }
            else{
            int j, seat_lt;
                for(j = req_src[i]; j > req_dst[i]; j--){
                    seat_lt = cap[25 * trn + cls]; 
                    if(track[1250 * trn + 50 * cls + src[trn] - j] + req_tkt[i] > seat_lt){
                        fail = 1; 
                        req_stat[i] = 0; 
                        atomicInc(&count[0], 5000); 
                        break; 
                    } 
                }
                if(!fail){
                    for(int j = req_src[i]; j > req_dst[i]; j--){
                        track[1250 * trn + 50 * cls + src[trn] - j] += req_tkt[i]; 
                    } 
                    req_stat[i] = 1; 
                    atomicInc(&count[1], 5000); 
                    atomicAdd(&count[2], req_tkt[i] * (req_src[i] - req_dst[i])); 
                }
                
            }
        }   
    }
}

int main(int argc,char **argv)
{
    int N; 
    scanf("%d", &N);   

    int *train_src = (int *) malloc (N * sizeof(int));
    int *train_dest = (int *) malloc (N * sizeof(int));
    int *capacities = (int *) malloc (N * 25 * sizeof(int));
    int *thread = (int *) malloc (N * 25 * sizeof(int));

    for(int i = 0; i < N; i++){
        int train_num;
        int num_classes;  
        
        scanf("%d", &train_num);
        scanf("%d", &num_classes);
        scanf("%d", &train_src[train_num]); 
        scanf("%d", &train_dest[train_num]);  
        
        for(int j = 0; j < num_classes; j++){
            int train_class;
            scanf("%d", &train_class);
            scanf("%d", &capacities[25 * train_num + train_class]);
        }

    }

    int *d_dest, *d_src, *d_cap; 
    cudaMalloc(&d_dest, N * sizeof(int));
	cudaMalloc(&d_src, N * sizeof(int));
	cudaMalloc(&d_cap, N * 25 * sizeof(int));
    cudaMemcpyAsync(d_dest, train_dest, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_src, train_src, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_cap, capacities, N * 25 * sizeof(int), cudaMemcpyHostToDevice);
    
    int num_batches; 
    scanf("%d", &num_batches); 

    int *req_train_num = (int *) malloc (5000 * sizeof(int));
    int *req_class_num = (int *) malloc (5000 * sizeof(int));
    int *req_src = (int *) malloc (5000 * sizeof(int));
    int *req_dest = (int *) malloc (5000 * sizeof(int));
    int *req_seats = (int *) malloc (5000 * sizeof(int));
    int *map = (int *) malloc (5000 * sizeof(int));

    int *stat = (int *) malloc (5000 * sizeof(int)); 
    int count[3]; 

    int *dreq_tr_num, *dreq_src, *dreq_dest, *dreq_seats, *dreq_cls, *dmap;  
    cudaMalloc(&dreq_tr_num, 5000 * sizeof(int));
    cudaMalloc(&dreq_src, 5000 * sizeof(int));
    cudaMalloc(&dreq_dest, 5000 * sizeof(int));
    cudaMalloc(&dreq_seats, 5000 * sizeof(int));
    cudaMalloc(&dreq_cls, 5000 * sizeof(int));
    cudaMalloc(&dmap, 5000 * sizeof(int));


    unsigned int *d_stat, *d_count;
    cudaMalloc(&d_stat, 5000 * sizeof(int));
	cudaMalloc(&d_count, 3 * sizeof(int));
    cudaMemcpyAsync(d_stat, stat, 5000 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_count, count, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int *d_track; 
    cudaMalloc(&d_track, N * 25 * 50 * sizeof(int)); 
    cudaMemset(d_track, 0, N * 25 * 50 * sizeof(int));
    

    int batch_size; 
    int req_id;


    for (int i = 0; i < num_batches; i++){
        memset(thread, -1, N * 25 * sizeof(int));

        scanf("%d", &batch_size);
        for(int j = 0; j < batch_size; j++){
            scanf("%d", &req_id);
            scanf("%d", &req_train_num[req_id]);
            scanf("%d", &req_class_num[req_id]);
            scanf("%d", &req_src[req_id]);
            scanf("%d", &req_dest[req_id]);
            scanf("%d", &req_seats[req_id]);

            int trn = req_train_num[req_id]; 
            int cls = req_class_num[req_id]; 
            
            if(thread[25 * trn + cls] == -1){
              thread[25 * trn + cls] = req_id;
              map[req_id] = req_id; 
            }
            else
              map[req_id] =  thread[25 * trn + cls]; 
        }

        cudaMemcpyAsync(dreq_tr_num, req_train_num, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dreq_src, req_src, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dreq_dest, req_dest, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dreq_seats, req_seats, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dreq_cls, req_class_num, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dmap, map, batch_size * sizeof(int), cudaMemcpyHostToDevice); 

        cudaMemset(d_stat, 0, 5000 * sizeof(int));
        cudaMemset(d_count, 0, 3 * sizeof(int));
        
        cudaDeviceSynchronize();

		int gridDimx = ceil(float(batch_size)/1024);
		int blockDimx = 1024; 
		
		batch_process<<<gridDimx, blockDimx>>>(
        batch_size, dreq_src, dreq_dest, dreq_seats, dreq_tr_num, dreq_cls,
        d_track, d_dest, d_src, d_cap, d_stat, d_count, dmap); 

        cudaMemcpy(stat, d_stat, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(count, d_count, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        
        for(int j = 0; j < batch_size; j++ ){
            if(stat[j])
              printf("success\n");
            else 
              printf("failure\n");
        } 
        printf("%d %d\n", count[1], count[0]);
        printf("%d\n", count[2]);
    }

    cudaFree(d_dest);
	cudaFree(d_src);
	cudaFree(d_cap);
    cudaFree(dreq_tr_num);
    cudaFree(dreq_src);
    cudaFree(dreq_dest);
    cudaFree(dreq_seats);
    cudaFree(dreq_cls);
    cudaFree(d_stat);
    cudaFree(d_count);
    cudaFree(d_track);

    free(train_src); 
    free(train_dest); 
    free(capacities);
    free(req_train_num);
    free(req_class_num);
    free(req_src);
    free(req_dest);
    free(req_seats); 
    free(stat);  
    free(thread);

    return 0;
}
