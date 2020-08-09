#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_FRONTIER_SIZE 16
#define BLOCK_QUEUE_SIZE 16
#define BLOCK_SIZE 8.0

//#define CudaCatchError(call) { \
//cudaError_t err = call; \
//if (err != cudaSuccess) { \
//printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
//exit(err); \
//} \
//}

#include "error_checker.h"

__global__ void frontier_init_kernel(int* p_frontier_tail_d, int* c_frontier_tail_d, int* p_frontier_d, int* visited_d, int* label_d, int source) {
  visited_d[source] = 1;
  *c_frontier_tail_d = 0;
  p_frontier_d[0] = source;
  *p_frontier_tail_d = 1;
  label_d[source] = 0;
}

__global__ void frontier_tail_swap_kernel(int* p_frontier_tail_d, int* c_frontier_tail_d) {
  
  *p_frontier_tail_d = *c_frontier_tail_d; 
  *c_frontier_tail_d = 0;
}

__global__ void BFS_Bqueue_kernel(int* p_frontier, int* p_frontier_tail, int* c_frontier,
    int* c_frontier_tail, int* edges, int* dest, int* label, int* visited) {

  __shared__ int c_frontier_s[BLOCK_QUEUE_SIZE];
  __shared__ int c_frontier_tail_s;
  __shared__ int our_c_frontier_tail;

  if (threadIdx.x == 0) {
    c_frontier_tail_s = 0;
  }
  __syncthreads();

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < *p_frontier_tail) {
    const int my_vertex = p_frontier[tid];
    for (int i = edges[my_vertex]; i < edges[my_vertex + 1]; ++i) {
      const int was_visited = atomicExch(&(visited[dest[i]]), 1);
      if (not was_visited) {
        label[dest[i]] = label[my_vertex] + 1;
        const int my_tail = atomicAdd(&c_frontier_tail_s, 1);
        if (my_tail < BLOCK_QUEUE_SIZE) {
          c_frontier_s[my_tail] = dest[i];
        } else {
          c_frontier_tail_s = BLOCK_QUEUE_SIZE;
          const int my_global_tail = atomicAdd(c_frontier_tail, 1);
          c_frontier[my_global_tail] = dest[i];
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < c_frontier_tail_s; i+= blockDim.x) {
      c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
    }
  }
}

void BFS_host(int source, int *edges, int *dest, int *label, int edges_num, int vertex_num){

  int *edges_d, *dest_d, *label_d, *visited_d;

  CudaCatchError(cudaMallocManaged((void**)&edges_d, (vertex_num + 1) * sizeof(int)));
  CudaCatchError(cudaMallocManaged((void**)&dest_d, edges_num * sizeof(int)));
  CudaCatchError(cudaMallocManaged((void**)&label_d, vertex_num * sizeof(int)));
  CudaCatchError(cudaMallocManaged((void**)&visited_d, vertex_num * sizeof(int)));

  CudaCatchError(cudaMemcpy(edges_d, edges, (vertex_num + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CudaCatchError(cudaMemcpy(dest_d, dest, edges_num * sizeof(int), cudaMemcpyHostToDevice));
  CudaCatchError(cudaMemcpy(label_d, label, vertex_num * sizeof(int), cudaMemcpyHostToDevice));

  int *frontier_d, *c_frontier_tail_d, *p_frontier_tail_d;
  CudaCatchError(cudaMallocManaged((void**)&frontier_d, 2 * MAX_FRONTIER_SIZE * sizeof(int)));
  CudaCatchError(cudaMallocManaged((void**)&c_frontier_tail_d, sizeof(int)));
  CudaCatchError(cudaMallocManaged((void**)&p_frontier_tail_d, sizeof(int)));
  int *c_frontier_d = frontier_d;
  int *p_frontier_d = frontier_d + MAX_FRONTIER_SIZE;

  CudaCatchError(cudaMemset(visited_d, 0, vertex_num * sizeof(int)));
  printf("1\n");
  CudaCatchError(cudaGetLastError());
  frontier_init_kernel <<< 1,1 >>>(p_frontier_tail_d, c_frontier_tail_d, p_frontier_d, visited_d, label_d, source);
  CudaCatchError(cudaGetLastError());
  int p_frontier_tail = 1;
  printf("2\n");
  while (p_frontier_tail > 0) {
    int num_blocks = ceil(p_frontier_tail / BLOCK_SIZE);
    BFS_Bqueue_kernel <<< num_blocks, BLOCK_SIZE >>> (p_frontier_d, p_frontier_tail_d, c_frontier_d, c_frontier_tail_d, edges_d, dest_d, label_d, visited_d);
    CudaCatchError(cudaGetLastError());
    CudaCatchError(cudaMemcpy(&p_frontier_tail, c_frontier_tail_d, sizeof(int), cudaMemcpyDeviceToHost));
    int *temp = c_frontier_d; c_frontier_d = p_frontier_d; p_frontier_d = temp;
    printf("3\n");
    frontier_tail_swap_kernel <<< 1,1 >>>(p_frontier_tail_d, c_frontier_tail_d);
    CudaCatchError(cudaGetLastError());
    printf("4\n");
  }
  cudaMemcpy(label, label_d, vertex_num * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(edges_d);
  cudaFree(dest_d);
  cudaFree(label_d);
  cudaFree(visited_d);
}



int main(int argc, char* argv[]) {

  int _edges[] = {0, 2, 4, 7, 9, 11, 12, 13, 15, 15};
  int _dest[] = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6};
  int _label[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  
  int* edges  = _edges;
  int* dest  = _dest;
  int* label  = _label;
  int source = 0;
  int edge_num = 15;
  int vertex_num = 9;
  BFS_host(source, edges, dest, label, edge_num, vertex_num);

  for (int i = 0; i < 9; i++) {
	printf("%d ", label[i]);
  }
  return 1;
}


