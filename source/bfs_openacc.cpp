//
// Created by Daniel Simon on 8/5/20.
//

#include "bfs_openacc.hpp"

#include <algorithm>
#include <utility>

std::vector<int> BFS_OpenACC(const Graph& graph, Graph::index_type source) {
  static constexpr int unreached = -1;
  const auto* row_indices = graph.matrix.row_indices.data();
  const auto* column_indices = graph.matrix.column_indices.data();
  const std::size_t num_nodes = graph.NumNodes();

  std::vector<int> distances(num_nodes, unreached);
  distances.at(source) = 0;
  auto* distances_alias = distances.data();

  auto* current_frontier = new Graph::index_type[num_nodes];
  std::size_t current_frontier_size = 0;
  auto* previous_frontier = new Graph::index_type[num_nodes];
  previous_frontier[0] = source;
  std::size_t previous_frontier_size = 1;

  #pragma acc data copyin(row_indices[:num_nodes], column_indices[:num_nodes]) \
        copy(distances_alias[:num_nodes], previous_frontier[:num_nodes]), create(current_frontier[:num_nodes])
  for (int k = 1; previous_frontier_size > 0; k++) {
    #pragma acc kernel device_type(nvidia)
    for (const Graph::index_type* row_ptr = previous_frontier;
         row_ptr < previous_frontier + previous_frontier_size; row_ptr++) {
      for (const Graph::index_type* col_ptr = column_indices + row_indices[*row_ptr];
           col_ptr < column_indices + row_indices[*row_ptr + 1]; col_ptr++) {
        if (distances_alias[*col_ptr] == unreached) {
          distances_alias[*col_ptr] = k;
          #pragma acc atomic write
          current_frontier[current_frontier_size] = *col_ptr;
          #pragma acc atomic update
          current_frontier_size++;
        }
      }
    }
    std::swap(previous_frontier, current_frontier);
    previous_frontier_size = current_frontier_size;
    current_frontier_size = 0;
  }
  return distances;
}