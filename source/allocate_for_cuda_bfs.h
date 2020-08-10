#ifndef CS535_GRAPHSEARCH_ALLOCATE_FOR_CUDA_BFS_H
#define CS535_GRAPHSEARCH_ALLOCATE_FOR_CUDA_BFS_H

#ifdef __cplusplus
extern "C" {
#endif

void AllocateAndCopyFor_device_BFS(int num_nodes, int num_edges, int source, const int* host_edges,
    const int* host_dests, int** device_edges, int** device_dests, int** device_label, int** device_visited,
    int** current_frontier, int** current_frontier_tail, int** previous_frontier, int** previous_frontier_tail);

void AllocateAndCopyFor_unified_BFS(int num_nodes, int num_edges, int source, const int* host_edges,
    const int* host_dests, int** device_edges, int** device_dests, int** device_label, int** device_visited,
    int** current_frontier, int** current_frontier_tail, int** previous_frontier, int** previous_frontier_tail);

void DeallocateFrom_device_BFS(int num_nodes, int* host_labels, int* device_edges, int* device_dests,
    int* device_labels, int* device_visited, int* current_frontier, int* current_frontier_tail,
    int* previous_frontier, int* previous_frontier_tail);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //CS535_GRAPHSEARCH_ALLOCATE_FOR_CUDA_BFS_H
