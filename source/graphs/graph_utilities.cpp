//
// Created by Daniel Simon on 8/5/20.
//

#include "graph_utilities.hpp"

#include <cassert>
#include <random>

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

Graph RandomGraphWithDiameter(std::size_t num_nodes, float average_numer_of_edges, float standard_deviation_of_edges) {
  static std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<std::size_t> node_distirbution{0, num_nodes - 1};
  std::normal_distribution<float> diameter_distirbution{average_numer_of_edges, standard_deviation_of_edges};
  Graph graph;
  for (std::size_t i = 0; i < num_nodes; i++) {
    for (std::size_t j = 0; j < (std::size_t)std::ceil(diameter_distirbution(generator)); j++) {
      bool added_successfully;
      do {
        // Add edge will return false if the two random numbers are the same or are already connected.
        added_successfully = graph.AddEdge(i, node_distirbution(generator));
      } while (not added_successfully);
    }
  }
  // Call resize to make sure the edge list is large enough so accessing any node wont go out of range.
  assert(not graph.Resize(num_nodes));
  return graph;
}