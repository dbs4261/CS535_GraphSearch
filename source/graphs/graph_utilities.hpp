//
// Created by Daniel Simon on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_GRAPH_UTILITIES_HPP
#define CS535_GRAPHSEARCH_GRAPH_UTILITIES_HPP

#include <utility>

#include "graph.hpp"

Graph RandomGraph(std::size_t num_nodes, std::size_t num_edges);

Graph RandomGraphWithDiameter(std::size_t num_nodes, float average_numer_of_edges, float standard_deviation_of_edges);

#endif //CS535_GRAPHSEARCH_GRAPH_UTILITIES_HPP
