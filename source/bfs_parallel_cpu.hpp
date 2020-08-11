//
// Converted from Grant's implementation in graph.c on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_BFS_PARALLEL_CPU_HPP
#define CS535_GRAPHSEARCH_BFS_PARALLEL_CPU_HPP

#include <vector>

#include "source/graphs/graph.hpp"

std::vector<int> BFS_ParallelCPU(const Graph& graph, Graph::index_type source);

#endif //CS535_GRAPHSEARCH_BFS_PARALLEL_CPU_HPP
