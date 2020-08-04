//
// Converted from Grant's implementation in graph.c on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_BFS_SEQUENTIAL_HPP
#define CS535_GRAPHSEARCH_BFS_SEQUENTIAL_HPP

#include <algorithm>
#include <vector>

#include "graph.hpp"

std::vector<int> BFS_sequential(const Graph& graph, Graph::index_type source) {
  static constexpr int unreached = -1;
  std::vector<Graph::index_type> previous_frontier = {source};
  std::vector<Graph::index_type> current_frontier;
  std::vector<int> distances(graph.NumNodes(), unreached);
  distances.at(source) = 0;

  while (not previous_frontier.empty()) {
    for (Graph::index_type frontier_node: previous_frontier) {
      // With ranges this could just be a view.
      auto begin = graph.matrix.column_indices.begin() + graph.matrix.row_indices.at(frontier_node);
      auto end = graph.matrix.column_indices.begin() + graph.matrix.row_indices.at(frontier_node + 1);
      std::copy_if(begin, end, std::back_inserter(current_frontier),
          [&distances](Graph::index_type node){
        return distances.at(node) == unreached;
      });
      auto frontier_distance = distances.at(frontier_node);
      for (Graph::index_type connected: current_frontier) {
        distances.at(connected) = frontier_distance + 1;
      }
    }
    // Swap instead of move to keep allocated memory
    std::swap(previous_frontier, current_frontier);
    current_frontier.clear();
  }

  return distances;
}

#endif //CS535_GRAPHSEARCH_BFS_SEQUENTIAL_HPP
