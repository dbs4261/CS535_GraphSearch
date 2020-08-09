//
// Created by gnbpdx on 8/6/20.
//

#ifndef CS535_GRAPHSEARCH_CUDA_REGULAR_BFS_H
#define CS535_GRAPHSEARCH_CUDA_REGULAR_BFS_H

#ifdef __cplusplus
extern "C" {
#endif

void Run_device_BFS(int num_nodes, int num_edges, int source, int* host_edges, int* host_dests, int* host_labels);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //CS535_GRAPHSEARCH_CUDA_REGULAR_BFS_H
