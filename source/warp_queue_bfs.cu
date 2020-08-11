#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define BLOCK_QUEUE_SIZE 16
#define BLOCK_SIZE 8.0
#define NUM_SUB_QUEUE 4
	
//#define CudaCatchError(call) { \
//cudaError_t err = call; \
//if (err != cudaSuccess) { \
//printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
//exit(err); \
//} \
//}

#include "error_checker.h"
#include "allocate_for_cuda_bfs.h"

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
  int subQueueIndex = threadIdx.x & (NUM_SUB_QUEUE - 1);
  if (subQueueIndex < NUM_SUB_QUEUE) {
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
  __syncwarp();
}

extern "C" void LaunchWarpQueueBFS_host(int num_nodes, int* edges, int* dests, int* labels, int* visited,
    int *c_frontier_tail, int *c_frontier, int *p_frontier_tail, int *p_frontier) {
  int frontier_size = 1;
  while (frontier_size > 0) {
    int num_blocks = ceil(frontier_size / BLOCK_SIZE);
    BFS_Bqueue_kernel <<< num_blocks, BLOCK_SIZE >>> (p_frontier, p_frontier_tail, c_frontier, c_frontier_tail, edges, dests, labels, visited);
    CudaCatchError(cudaGetLastError());
    CudaCatchError(cudaMemcpy(&frontier_size, c_frontier_tail, sizeof(int), cudaMemcpyDeviceToHost));
    int *temp = c_frontier; c_frontier = p_frontier; p_frontier = temp;
    frontier_tail_swap_kernel<<< 1,1 >>>(p_frontier_tail, c_frontier_tail);
    CudaCatchError(cudaGetLastError());
  }
}

extern "C" void Run_WarpQueue_BFS(int num_nodes, int num_edges, int source, int* host_edges, int* host_dests, int* host_labels) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  AllocateAndCopyFor_device_BFS(num_nodes, num_edges, source, host_edges, host_dests, &device_edges, &device_dests, &device_labels, &device_visited, &current_frontier, &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
  LaunchWarpQueueBFS_host(num_nodes, device_edges, device_dests, device_labels, device_visited, current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
  DeallocateFrom_device_BFS(num_nodes, host_labels, device_edges, device_dests, device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier, previous_frontier_tail);
}

//int main(int argc, char* argv[]) {
//
//  int _edges[] = {0, 2, 4, 7, 9, 11, 12, 13, 15, 15};
//  int _dest[] = {1, 2, 3, 4, 5, 6, 7, 4, 8, 5, 8, 6, 8, 0, 6};
//  int _label[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
//
//  int* edges  = _edges;
//  int* dest  = _dest;
//  int* label  = _label;
//  int source = 0;
//  int edge_num = 15;
//  int vertex_num = 9;
//  Run_WarpQueue_BFS(edge_num, vertex_num, source, edges, dest, label);
//
//  for (int i = 0; i < 9; i++) {
//	printf("%d ", label[i]);
//  }
//  return EXIT_SUCCESS;
//}
