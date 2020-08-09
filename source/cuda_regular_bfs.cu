//
// Created by gnbpdx on 8/6/20.
//

#include <cuda.h>
#include "../applications/graph.c"
#include <stdlib.h>
__global__ void device_BFS(int source, int* edges, int* dest, int* label, int* visited, int *c_frontier_tail, int *c_frontier, int *p_frontier_tail, int *p_frontier)
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
	int* h_edges, *h_dest, *h_data, *d_edges, *d_dest;
	int** graph = init_graph_adjacency(argv[1], &num_nodes);
	num_edges = calc_num_edges(graph, num_nodes);
	h_dest = (int*)malloc(sizeof(int)*num_edges);
	h_edges = (int*)malloc(sizeof(int)*(num_nodes+1));
	h_data = (int*)malloc(sizeof(int)* num_edges);
	cudaMalloc(&d_dest, sizeof(int)*num_edges);
	cudaMalloc(&d_edges, sizeof(int)*(num_nodes + 1));	
  convert_graph_to_csr(graph, h_edges, h_dest, h_data, num_nodes, num_edges);
	cudaMemcpy(d_edges, h_edges, sizeof(int)*(num_nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dest, h_dest, sizeof(int)*num_edges, cudaMemcpyHostToDevice);
  int* h_label, *d_label, *h_visited, *d_visited;
	h_label = (int*)malloc(sizeof(int)*num_nodes);
	h_visited = (int*)malloc(sizeof(int)*num_nodes);
	cudaMalloc(&d_label, sizeof(int)*num_nodes);
	cudaMalloc(&d_visited, sizeof(int)*num_nodes);
	int* d_frontier, *h_frontier;
	h_frontier = (int*)malloc(sizeof(int)*2*num_nodes);
	cudaMalloc(&d_frontier, sizeof(int)*2*num_nodes);
	int* c_frontier = d_frontier;
	int* p_frontier = d_frontier + (sizeof(int) * num_nodes);
	int* h_p_frontier = h_frontier + sizeof(int) * num_nodes;
	int* p_frontier_tail, *c_frontier_tail;
	int h_p_frontier_tail, h_c_frontier_tail;
	cudaMalloc(&p_frontier_tail, sizeof(int));
	cudaMalloc(&c_frontier_tail, sizeof(int));
	h_c_frontier_tail = 0;
	h_p_frontier_tail = 1;
	cudaMemcpy(c_frontier_tail, &h_c_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(p_frontier_tail, &h_p_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);	
	
	for (int i = 0; i < num_nodes; ++i)
	{
		h_label[i] = -1;
		h_visited[i] = 0;
	}
	h_visited[source] = 1;
	h_p_frontier[0] = source;
	h_label[source] = 0;
	cudaMemcpy(d_label, h_label, sizeof(int)*num_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_visited, h_visited, sizeof(int)*num_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frontier, h_frontier, sizeof(int)*2*num_nodes, cudaMemcpyHostToDevice);
	int count = 0;
	while(h_p_frontier_tail > 0) 
	{	
		int block_size = 32;
		int grid_size = ceil(h_p_frontier_tail/32.0);
		device_BFS<<<grid_size, block_size>>>(source, d_edges, d_dest, d_label, d_visited, c_frontier_tail, c_frontier, p_frontier_tail, p_frontier);
		cudaDeviceSynchronize();
		int* temp = c_frontier;
		c_frontier = p_frontier;
		p_frontier = temp;
		cudaMemcpy(&h_p_frontier_tail, p_frontier_tail, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_c_frontier_tail, c_frontier_tail, sizeof(int), cudaMemcpyDeviceToHost);
		h_p_frontier_tail = h_c_frontier_tail;
		h_c_frontier_tail = 0;
		cudaMemcpy(p_frontier_tail, &h_p_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(c_frontier_tail, &h_c_frontier_tail, sizeof(int), cudaMemcpyHostToDevice);
		count++;
	}
	cudaMemcpy(h_label, d_label, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost);
	print_label(0, h_label, num_nodes);	
	printf("%d", count);
}
