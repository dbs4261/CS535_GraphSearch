//
// Converted from Grant's implementation in graph.c on 8/2/20.
//

#include "bfs_parallel_stl.hpp"

#include <algorithm>
#include <execution>
#include <utility>
#include <vector>

std::vector<int> BFS_ParallelSTL(const Graph& graph, Graph::index_type source) {
  static constexpr int unreached = -1;
  std::vector<Graph::index_type> previous_frontier = {source};
  std::vector<Graph::index_type> current_frontier;
  std::vector<int> distances(graph.NumNodes(), unreached);
  distances.at(source) = 0;

  while (not previous_frontier.empty()) {
    std::for_each(std::execution::par_unseq, previous_frontier.begin(), previous_frontier.end(),
        [&current_frontier, &distances, graph](Graph::index_type frontier_node)->void{
      auto begin = graph.matrix.column_indices.begin() + graph.matrix.row_indices.at(frontier_node);
      auto end = graph.matrix.column_indices.begin() + graph.matrix.row_indices.at(frontier_node + 1);
      auto frontier_distance = distances.at(frontier_node);
      std::copy_if(begin, end, std::back_inserter(current_frontier),
          [&distances, frontier_distance](Graph::index_type node)->bool{
        if (distances.at(node) == unreached) {
          distances.at(node) = frontier_distance + 1;
          return true;
        }
        return false;
      });
    });
    // Swap instead of move to keep allocated memory
    std::swap(previous_frontier, current_frontier);
    current_frontier.clear();
  }

  return distances;
}