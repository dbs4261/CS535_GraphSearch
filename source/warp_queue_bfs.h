//
// Created by developer on 8/10/20.
//

#ifndef CS535_GRAPHSEARCH_WARP_QUEUE_BFS_H
#define CS535_GRAPHSEARCH_WARP_QUEUE_BFS_H

#ifdef __cplusplus
extern "C" {
#endif

void LaunchWarpQueueBFS_host(int num_nodes, int* edges, int* dests, int* labels, int* visited,
    int *c_frontier_tail, int *c_frontier, int *p_frontier_tail, int *p_frontier);

void Run_WarpQueue_BFS(int num_nodes, int num_edges, int source, int* host_edges, int* host_dests, int* host_labels);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //CS535_GRAPHSEARCH_WARP_QUEUE_BFS_H
