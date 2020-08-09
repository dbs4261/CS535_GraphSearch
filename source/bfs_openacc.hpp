//
// Created by Daniel Simon on 8/5/20.
//

#ifndef CS535_GRAPHSEARCH_BFS_OPENACC_H
#define CS535_GRAPHSEARCH_BFS_OPENACC_H

#include <vector>

#include "source/graphs/graph.hpp"

std::vector<int> BFS_OpenACC(const Graph& graph, Graph::index_type source);

#endif //CS535_GRAPHSEARCH_BFS_OPENACC_H
