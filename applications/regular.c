//
// Created by gnbpdx on 8/6/20.
//

#include <stdlib.h>

#include "source/graph.c"
#include "source/cuda_regular_bfs.h"

int main(int argc, char** argv) {
	int source = 0;
	int num_nodes;
	int num_edges;
	int* h_edges;
	int* h_dest;
	int* h_data;
	int* h_label;
	int** graph = init_graph_adjacency(argv[1], &num_nodes);
	num_edges = calc_num_edges(graph, num_nodes);
	h_dest = (int*)malloc(sizeof(int)*num_edges);
	h_edges = (int*)malloc(sizeof(int)*(num_nodes+1));
	h_data = (int*)malloc(sizeof(int)* num_edges);
	h_label = (int*)malloc(sizeof(int)* num_edges);
  convert_graph_to_csr(graph, h_edges, h_dest, h_data, num_nodes, num_edges);
  Run_device_BFS(num_nodes, num_edges, source, h_edges, h_dest, h_label);
	print_label(0, h_label, num_nodes);
	int count = 0;
	for (int i = 0; i < num_nodes; i++) {
    count = h_label[i] > count ? h_label[i] : count;
	}
	printf("count: %d\n", count);
}
