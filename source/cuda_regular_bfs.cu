//
// Created by gnbpdx on 8/6/20.
//

#include <cuda.h>
#include "graph.c"
#include <stdlib.h>

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

void AllocateAndCopyFor_device_BFS(int num_nodes, int num_edges, int source, const int* host_edges,
    const int* host_dests, int** device_edges, int** device_dests, int** device_label, int** device_visited,
    int** current_frontier, int** current_frontier_tail, int** previous_frontier, int** previous_frontier_tail) {
  CudaCatchError(cudaMalloc(device_edges, sizeof(int) * (num_nodes + 1)));
  CudaCatchError(cudaMemcpy(*device_edges, host_edges, sizeof(int) * (num_nodes + 1), cudaMemcpyHostToDevice));
  CudaCatchError(cudaMalloc(device_dests, sizeof(int) * num_edges));
  CudaCatchError(cudaMemcpy(*device_dests, host_dests, sizeof(int) * num_edges, cudaMemcpyHostToDevice));
  int* temp = (int*)malloc(sizeof(int) * num_nodes);
  CudaCatchError(cudaMalloc(device_label, sizeof(int) * num_nodes));
  for (int i = 0; i < num_nodes; i++) {
    temp[i] = -1;
  }
  temp[source] = 0;
  CudaCatchError(cudaMemcpy(*device_label, temp, sizeof(int) * num_nodes, cudaMemcpyHostToDevice));
  CudaCatchError(cudaMalloc(device_visited, sizeof(int) * num_nodes));
  for (int i = 0; i < num_nodes; i++) {
    temp[i] = 0;
  }
  temp[source] = 1;
  CudaCatchError(cudaMemcpy(*device_visited, temp, sizeof(int) * num_nodes, cudaMemcpyHostToDevice));
  CudaCatchError(cudaMalloc(previous_frontier, sizeof(int) * num_nodes));
  temp[0] = source;
  CudaCatchError(cudaMemcpy(*previous_frontier, temp, sizeof(int), cudaMemcpyHostToDevice));
  CudaCatchError(cudaMalloc(current_frontier, sizeof(int) * num_nodes));
  temp[0] = 1;
  temp[1] = 0;
  CudaCatchError(cudaMalloc(previous_frontier_tail, sizeof(int)));
  CudaCatchError(cudaMemcpy(*previous_frontier_tail, temp, sizeof(int), cudaMemcpyHostToDevice));
  CudaCatchError(cudaMalloc(current_frontier_tail, sizeof(int)));
  CudaCatchError(cudaMemcpy(*current_frontier_tail, temp + 1, sizeof(int), cudaMemcpyHostToDevice));
  free(temp);
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

void DeallocateFrom_device_BFS(int num_nodes, int* host_labels, int* device_edges, int* device_dests,
    int* device_labels, int* device_visited, int* current_frontier, int* current_frontier_tail,
    int* previous_frontier, int* previous_frontier_tail) {
  CudaCatchError(cudaMemcpy(host_labels, device_labels, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost));
  CudaCatchError(cudaFree(device_edges));
  CudaCatchError(cudaFree(device_dests));
  CudaCatchError(cudaFree(device_labels));
  CudaCatchError(cudaFree(device_visited));
  // These are the same array so allocate the lower pointer which is the beginning
//  CudaCatchError(cudaFree(previous_frontier < current_frontier ? previous_frontier : current_frontier));
  CudaCatchError(cudaFree(previous_frontier));
  CudaCatchError(cudaFree(current_frontier));
//  CudaCatchError(cudaFree(previous_frontier_tail < current_frontier_tail ? previous_frontier_tail : current_frontier_tail));
  CudaCatchError(cudaFree(previous_frontier_tail));
  CudaCatchError(cudaFree(current_frontier_tail));
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
