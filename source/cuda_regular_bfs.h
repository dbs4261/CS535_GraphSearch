//
// Created by gnbpdx on 8/6/20.
//

#ifndef CS535_GRAPHSEARCH_CUDA_REGULAR_BFS_H
#define CS535_GRAPHSEARCH_CUDA_REGULAR_BFS_H

void Run_device_BFS(int num_nodes, int num_edges, int source, int* host_edges, int* host_dests, int* host_labels);

#endif //CS535_GRAPHSEARCH_CUDA_REGULAR_BFS_H
