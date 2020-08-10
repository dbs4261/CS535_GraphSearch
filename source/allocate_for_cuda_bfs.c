#include <cuda.h>

#include "allocate_for_cuda_bfs.h"
#include "error_checker.h"

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
  CudaCatchError(cudaDeviceSynchronize());
}

void AllocateAndCopyFor_unified_BFS(int num_nodes, int num_edges, int source, const int* host_edges,
    const int* host_dests, int** device_edges, int** device_dests, int** device_label, int** device_visited,
    int** current_frontier, int** current_frontier_tail, int** previous_frontier, int** previous_frontier_tail) {
  CudaCatchError(cudaMallocManaged(device_edges, sizeof(int) * (num_nodes + 1), cudaMemAttachGlobal));
  CudaCatchError(cudaMemcpy(*device_edges, host_edges, sizeof(int) * (num_nodes + 1), cudaMemcpyHostToDevice));
  CudaCatchError(cudaMallocManaged(device_dests, sizeof(int) * num_edges, cudaMemAttachGlobal));
  CudaCatchError(cudaMemcpy(*device_dests, host_dests, sizeof(int) * num_edges, cudaMemcpyHostToDevice));
  CudaCatchError(cudaMallocManaged(device_label, sizeof(int) * num_nodes, cudaMemAttachGlobal));
  for (int i = 0; i < num_nodes; i++) {
    (*device_label)[i] = -1;
  }
  (*device_label)[source] = 0;
  CudaCatchError(cudaMallocManaged(device_visited, sizeof(int) * num_nodes, cudaMemAttachGlobal));
  for (int i = 0; i < num_nodes; i++) {
    (*device_visited)[i] = 0;
  }
  (*device_visited)[source] = 1;
  CudaCatchError(cudaMallocManaged(previous_frontier, sizeof(int) * num_nodes, cudaMemAttachGlobal));
  (*previous_frontier)[0] = source;
  CudaCatchError(cudaMallocManaged(current_frontier, sizeof(int) * num_nodes, cudaMemAttachGlobal));
  CudaCatchError(cudaMallocManaged(previous_frontier_tail, sizeof(int), cudaMemAttachGlobal));
  (*previous_frontier_tail)[0] = 1;
  CudaCatchError(cudaMallocManaged(current_frontier_tail, sizeof(int), cudaMemAttachGlobal));
  (*current_frontier_tail)[0] = 0;
  CudaCatchError(cudaDeviceSynchronize());
}

void DeallocateFrom_device_BFS(int num_nodes, int* host_labels, int* device_edges, int* device_dests,
    int* device_labels, int* device_visited, int* current_frontier, int* current_frontier_tail,
    int* previous_frontier, int* previous_frontier_tail) {
  CudaCatchError(cudaMemcpy(host_labels, device_labels, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost));
  CudaCatchError(cudaFree(device_edges));
  CudaCatchError(cudaFree(device_dests));
  CudaCatchError(cudaFree(device_labels));
  CudaCatchError(cudaFree(device_visited));
  CudaCatchError(cudaFree(previous_frontier));
  CudaCatchError(cudaFree(current_frontier));
  CudaCatchError(cudaFree(previous_frontier_tail));
  CudaCatchError(cudaFree(current_frontier_tail));
  CudaCatchError(cudaDeviceSynchronize());
}

