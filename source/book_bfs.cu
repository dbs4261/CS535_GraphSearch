//
// Created by developer on 7/26/20.
//

#include "book_bfs.cuh"

namespace book {

__global__ void BFS_Bqueue_kernel(unsigned int* p_frontier, unsigned int* p_frontier_tail, unsigned int* c_frontier,
    unsigned int* c_frontier_tail, unsigned int* edges, unsigned int* dest, unsigned int* label, unsigned int* visited) {
  __shared__ unsigned int c_frontier_s[BLOCK_QUEUE_SIZE];
  __shared__ unsigned int c_frontier_tail_s;
  __shared__ unsigned int our_c_frontier_tail;

  if (threadIdx.x == 0) {
    c_frontier_tail_s = 0;
  }
  __syncthreads();

  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < *p_frontier_tail) {
    const unsigned int my_vertex = p_frontier[tid];
    for (unsigned int i = edges[my_vertex]; i < edges[my_vertex + 1]; ++i) {
      const unsigned int was_visited = atomicExch(&(visited[dest[i]]), 1);
      if (not was_visited) {
        label[dest[i]] = label[my_vertex];
        const unsigned int my_tail = atomicAdd(&c_frontier_tail_s, 1);
        if (my_tail < BLOCK_QUEUE_SIZE) {
          c_frontier_s[my_tail] = dest[i];
        } else {
          c_frontier_tail_s = BLOCK_QUEUE_SIZE;
          const unsigned int my_global_tail = atomicAdd(c_frontier_tail, 1);
          c_frontier[my_global_tail] = dest[i];
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
    }
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < c_frontier_tail_s; i+= blockDim.x) {
      c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
    }
  }
}

cudaError_t Launch_BFS_Bqueue_kernel(dim3 grid_size, dim3 block_size, unsigned int* p_frontier,
    unsigned int* p_frontier_tail, unsigned int* c_frontier, unsigned int* c_frontier_tail,
    unsigned int* edges, unsigned int* dest, unsigned int* label, unsigned int* visited) {
  BFS_Bqueue_kernel<<<grid_size, block_size>>>(p_frontier, p_frontier_tail,
      c_frontier, c_frontier_tail, edges, dest, label, visited);
  return cudaGetLastError();
}

}