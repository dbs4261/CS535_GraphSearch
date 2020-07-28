#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define BUFFER 1024
//initializes graph from file
int** init_graph_adjacency(char* filename, int* num_nodes) 
{
	FILE* FP = fopen(filename, "r");
	fscanf(FP, "%d", num_nodes);
	int** graph = (int**)malloc(sizeof(int*) * *num_nodes);
	for (int i = 0; i < *num_nodes; ++i) 
		graph[i] = (int*)malloc(sizeof(int) * *num_nodes);
	for (int i = 0; i < *num_nodes; ++i)
		for (int j = 0; j < *num_nodes; ++j)
			fscanf(FP, "%d", graph[i] + j);
	fclose(FP);
	return graph;
}
int** init_zero_graph(int num_nodes)
{
	int** graph = (int**)malloc(sizeof(int*) * num_nodes);
	for (int i = 0; i < num_nodes; ++i) 
		graph[i] = (int*)malloc(sizeof(int) * num_nodes);
	for (int i = 0; i < num_nodes; ++i)
		for (int j = 0; j < num_nodes; ++j)
			graph[i][j] = 0;
	return graph;
}
//prints out graph to console
void print_graph(int** graph, int num_nodes)
{
	for (int i = 0; i < num_nodes; ++i)
	{
		for (int j = 0; j < num_nodes; ++j)
		{
			printf("%d ", graph[i][j]);
		}
		puts("");
	}

}
void print_graph_to_file(int** graph, int num_nodes, char* filename)
{
	FILE* FP = fopen(filename, "w");
	fprintf(FP, "%d\n", num_nodes);
	for (int i = 0; i < num_nodes; ++i)
	{
		for (int j = 0; j < num_nodes; ++j)
		{
			fprintf(FP, "%d ", graph[i][j]);
		}
		fprintf(FP, "\n");
	}
	fclose(FP);
}
void convert_graph_to_csr(int** graph, int* edges, int* dest, int* data, int num_nodes, int num_edges)
{
	int index = 0;
	//fill arrays
	edges[0] = 0;
	for (int i = 0; i < num_nodes; ++i)
	{
		for (int j = 0; j < num_nodes; ++j)
		{
			if(graph[i][j] != 0)
			{
				data[index] = graph[i][j];
				dest[index] = j;	
				++index;
			}
		}
		edges[i+1] = index;
	}
}
int calc_num_edges(int** graph, int num_nodes)
{

	//count number of non-zero edges
	int num_edges = 0;
	for (int i = 0; i < num_nodes; ++i)
		for (int j = 0; j < num_nodes; ++j)
			if (graph[i][j] != 0)
				++num_edges;
	return num_edges;
}
void print_csr(int* edges, int* dest, int* data, int num_nodes, int num_edges)
{
	printf("Data array:\n");
	for (int i = 0; i < num_edges; ++i)
		printf("%d ", *(data + i));
	printf("\nDest array:\n");
	for (int i = 0; i < num_edges; ++i)
		printf("%d ", *(dest + i));
	printf("\nEdges array:\n");
	for (int i = 0; i < (num_nodes + 1); ++i)
		printf("%d ", *(edges + i));
	printf("\n");


}
void print_label(int source, int* label, int num_nodes)
{
	printf("Distance from Node %d:\n", source);
	for (int i = 0; i < num_nodes; ++i)
		printf("%d ", label[i]);
	printf("\n");
}

void BFS_sequential(int source, int * edges, int *dest, int *label, int num_nodes)
{
	int MAX_FRONTIER_SIZE = num_nodes;
	int frontier[2][MAX_FRONTIER_SIZE];
	int *c_frontier = frontier[0];
	int c_frontier_tail = 0;
	int *p_frontier = frontier[1];
	int p_frontier_tail = 1;
	int c_vertex;
	for (int i = 0; i < num_nodes; ++i)
		label[i] = -1;

	p_frontier[0] = source;
	label[source] = 0;
	while (p_frontier_tail > 0)
	{
		for (int f = 0; f < p_frontier_tail; f++)
		{
			c_vertex = p_frontier[f];
			for (int i = edges[c_vertex]; i < edges[c_vertex+1]; i++)
			{
				if (label[dest[i]] == -1)
				{
					c_frontier[c_frontier_tail] = dest[i];
					++c_frontier_tail;
					label[dest[i]] = label[c_vertex] + 1;
				}
			}
		}
		int* temp = c_frontier;
		c_frontier = p_frontier;
		p_frontier = temp;
		p_frontier_tail = c_frontier_tail;
		c_frontier_tail = 0;
	}
}
void connected_add_edge(int** graph, int vertex1, int vertex2, int data)
{
	graph[vertex1][vertex2] = data;
	graph[vertex2][vertex1] = data;
}
//randomly adds edges until graph is connected
//Assumes graph is initialized with a 0 adjacency matrix
//Since the graph stops making edges once it is connected this will tend to make a sparse graph
void create_random_simple_connected_graph(int num_nodes, int** graph, int max_data)
{
	bool connected = false;
	int* edges = (int*)malloc(sizeof(int) * (num_nodes + 1));
	int* dest = malloc(sizeof(int) * num_nodes * num_nodes);
	int* data = malloc(sizeof(int) * num_nodes * num_nodes);
	int num_edges = 0;
	int *label = malloc(sizeof(int)*num_nodes);
	while(!connected)
	{
		int edge1 = rand() % num_nodes;
		int edge2 = rand() % num_nodes;
		int datapoint = rand() % max_data + 1;
		if ((edge1 == edge2) || (graph[edge1][edge2] != 0)) 
			continue;
		connected_add_edge(graph, edge1, edge2, datapoint);
		++num_edges;
		//convert graph to csr
		convert_graph_to_csr(graph, edges, dest, data, num_nodes, num_edges);
		//Perform BFS on csr
		BFS_sequential(0, edges, dest, label, num_nodes);
		//Determine if graph is connected
		connected = true;
		for (int i = 0; i < num_nodes; ++i)
			if (label[i] == -1)
			{
				connected = false; 
				break;
			}	
	}
	free(edges);
	free(dest);
	free(data);
	free(label);
}
//First argument in argv is number of nodes
//Second argument is filename to save graph in
int main(int argc, char** argv) {
  assert(argc == 3);
  	srand(time(NULL));
	int num_nodes, num_edges;
	int* edges, *dest, *data;
//	int** graph = init_graph_adjacency(argv[1], &num_nodes);
	num_nodes = atoi(argv[1]);
	int** graph = init_zero_graph(num_nodes);
	create_random_simple_connected_graph(num_nodes, graph, 100);
	num_edges = calc_num_edges(graph, num_nodes);
	data = (int*)malloc(sizeof(int)*num_edges);
	dest = (int*)malloc(sizeof(int)*num_edges);
	edges = (int*)malloc(sizeof(int)*(num_nodes + 1));
	print_graph(graph, num_nodes);
	convert_graph_to_csr(graph, edges, dest, data, num_nodes, num_edges);
	print_csr(edges, dest, data, num_nodes, num_edges);
	int* label = malloc(sizeof(int)*num_nodes);
	BFS_sequential(0, edges, dest, label, num_nodes);
	print_label(0, label, num_nodes);
	print_graph_to_file(graph, num_nodes, argv[2]);
	for(int i = 0; i < num_nodes; ++i)
		free(graph[i]);
	free(graph);
	free(data);
	free(dest);
	free(edges);
	free(label);
	return 0;
	
}
