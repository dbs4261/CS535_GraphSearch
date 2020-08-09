//
// Created by gnbpdx on 8/6/20.
//

#include <cuda.h>
#include "../applications/graph.c"

__global__ void BFS_UNIFIED(int source, int* edges, int* dest, int* label, int* visited, int *c_frontier_tail, int *c_frontier, int *p_frontier_tail, int *p_frontier)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *p_frontier_tail)
	{
		int c_vertex = p_frontier[i];
		for (int i = edges[c_vertex]; i < edges[c_vertex+1]; ++i)
		{
			int was_visited = atomicExch(visited + dest[i], 1);
			if (!was_visited)
			{
				int old_tail = atomicAdd(c_frontier_tail, 1);
				c_frontier[old_tail] = dest[i];
				label[dest[i]] = label[c_vertex] + 1;
			}
		}
	
	}	
}

int main(int argc, char** argv)
{
	int source = 0;
	int num_nodes, num_edges;
	int* edges, *dest, *data;
	int** graph = init_graph_adjacency(argv[1], &num_nodes);
	num_edges = calc_num_edges(graph, num_nodes);
	cudaMallocManaged(&data, sizeof(int)*num_edges);
	cudaMallocManaged(&dest, sizeof(int)*num_edges);
	cudaMallocManaged(&edges, sizeof(int)*(num_nodes + 1));	
  convert_graph_to_csr(graph, edges, dest, data, num_nodes, num_edges);
  int* label, *visited;
	cudaMallocManaged(&label, sizeof(int)*num_nodes);
	cudaMallocManaged(&visited, sizeof(int)*num_nodes);
	int* frontier;
	cudaMallocManaged(&frontier, sizeof(int)*2*num_nodes);
	int* c_frontier = frontier;
	int* p_frontier = frontier + (sizeof(int) * num_nodes);
	int* p_frontier_tail, *c_frontier_tail;
	cudaMallocManaged(&p_frontier_tail, sizeof(int));
	cudaMallocManaged(&c_frontier_tail, sizeof(int));
	*c_frontier_tail = 0;
	*p_frontier_tail = 1;
	for (int i = 0; i < num_nodes; ++i)
	{
		label[i] = -1;
		visited[i] = 0;
	}
	visited[source] = 1;
	p_frontier[0] = source;
	label[source] = 0;
	while(*p_frontier_tail > 0) 
	{	
		int block_size = 32;
		int grid_size = ceil(*p_frontier_tail/32.0);
		BFS_UNIFIED<<<grid_size, block_size>>>(source, edges, dest, label, visited, c_frontier_tail, c_frontier, p_frontier_tail, p_frontier);
		cudaDeviceSynchronize();
		int* temp = c_frontier;
		c_frontier = p_frontier;
		p_frontier = temp;
		*p_frontier_tail = *c_frontier_tail;
		*c_frontier_tail = 0;
	}
	print_label(0, label, num_nodes);	

}
