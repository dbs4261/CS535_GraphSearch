//
// Created by gnbpdx on 8/6/20.
//

#include <cuda.h>
#include "graph.c"
#include <stdlib.h>

#include "allocate_for_cuda_bfs.h"
#include "error_checker.h"

__global__ void device_BFS(const int* edges, const int* dests, int* labels, int* visited, int* c_frontier_tail, int* c_frontier, int* p_frontier_tail, int* p_frontier) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *p_frontier_tail) {
		int c_vertex = p_frontier[i];
		for (int i = edges[c_vertex]; i < edges[c_vertex+1]; i++) {
			int was_visited = atomicExch(visited + dests[i], 1);
			if (!was_visited) {
				int old_tail = atomicAdd(c_frontier_tail, 1);
				c_frontier[old_tail] = dests[i];
				labels[dests[i]] = labels[c_vertex] + 1;
			}
		}
	}	
}

void Launch_device_BFS(int num_nodes, int* edges, int* dests, int* labels, int* visited, int *c_frontier_tail, int *c_frontier, int *p_frontier_tail, int *p_frontier) {
  unsigned int block_size = 32;
  unsigned int grid_size = ceil(num_nodes/(float)block_size);
  int frontier_size = 1;
  int* pointer_swap = nullptr;
  while (frontier_size > 0) {
    device_BFS<<<grid_size, block_size>>>(edges, dests, labels, visited, c_frontier_tail, c_frontier, p_frontier_tail, p_frontier);
    CudaCatchError(cudaGetLastError());
    CudaCatchError(cudaDeviceSynchronize());
    pointer_swap = p_frontier;
    p_frontier = c_frontier;
    c_frontier = pointer_swap;
    pointer_swap = p_frontier_tail;
    p_frontier_tail = c_frontier_tail;
    c_frontier_tail = pointer_swap;
    frontier_size = 0;
    CudaCatchError(cudaMemcpy(c_frontier_tail, &frontier_size, sizeof(int), cudaMemcpyHostToDevice));
    CudaCatchError(cudaMemcpy(&frontier_size, p_frontier_tail, sizeof(int), cudaMemcpyDeviceToHost));
  }
}

extern "C" void Run_device_BFS(int num_nodes, int num_edges, int source, int* host_edges, int* host_dests, int* host_labels) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  AllocateAndCopyFor_device_BFS(num_nodes, num_edges, source, host_edges, host_dests, &device_edges, &device_dests, &device_labels, &device_visited, &current_frontier, &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
  Launch_device_BFS(num_nodes, device_edges, device_dests, device_labels, device_visited, current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
  DeallocateFrom_device_BFS(num_nodes, host_labels, device_edges, device_dests, device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier, previous_frontier_tail);
}
