#ifndef CS535_GRAPHSEARCH_GRAPH_H
#define CS535_GRAPHSEARCH_GRAPH_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define BUFFER 1024
//initializes graph from file
int** init_graph_adjacency(char* filename, int* num_nodes);

int** init_zero_graph(int num_nodes);

//prints out graph to console
void print_graph(int** graph, int num_nodes);

void print_graph_to_file(int** graph, int num_nodes, char* filename);

void convert_graph_to_csr(int** graph, int* edges, int* dest, int* data, int num_nodes, int num_edges);

int calc_num_edges(int** graph, int num_nodes);

void print_csr(int* edges, int* dest, int* data, int num_nodes, int num_edges);

void print_label(int source, int* label, int num_nodes);

void BFS_sequential(int source, int * edges, int *dest, int *label, int num_nodes);

void connected_add_edge(int** graph, int vertex1, int vertex2, int data);

//randomly adds edges until graph is connected
//Assumes graph is initialized with a 0 adjacency matrix
//Since the graph stops making edges once it is connected this will tend to make a sparse graph
void create_random_simple_connected_graph(int num_nodes, int** graph, int max_data);

#endif //CS535_GRAPHSEARCH_GRAPH_H
