//
// Created by Daniel Simon on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_GRAPH_UTILITIES_HPP
#define CS535_GRAPHSEARCH_GRAPH_UTILITIES_HPP

#include <cassert>
#include <random>
#include <utility>
#include <vector>

#include "graph.hpp"

Graph RandomGraph(std::size_t num_nodes, std::size_t num_edges) {
  static std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<std::size_t> distirbution{0, num_nodes  - 1};
  Graph graph;
  for (std::size_t i = 0; i < num_edges; i++) {
    bool added_successfully;
    do {
      // Add edge will return false if the two random numbers are the same or are already connected.
      added_successfully = graph.AddEdge(distirbution(generator), distirbution(generator));
    } while (not added_successfully);
  }
  // Call resize to make sure the edge list is large enough so accessing any node wont go out of range.
  assert(not graph.Resize(num_nodes));
  return graph;
}

#endif //CS535_GRAPHSEARCH_GRAPH_UTILITIES_HPP
