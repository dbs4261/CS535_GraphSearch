//
// Created by Daniel Simon on 8/5/20.
//

#include "bfs_openacc.hpp"

#include <algorithm>
#include <utility>

std::vector<int> BFS_OpenACC(const Graph& graph, Graph::index_type source) {
//  const Graph::index_type* row_indices = graph.matrix.row_indices.data();
//  const std::size_t row_indices_size = graph.matrix.row_indices.size();
//  const Graph::index_type* column_indices = graph.matrix.column_indices.data();
//  const std::size_t column_indices_size = graph.matrix.column_indices.size();
  const std::size_t num_nodes = graph.NumNodes();
  std::vector<int> distances(num_nodes, -1); // AKA L-vector
//  distances.at(source) = 0;
//  int* distances_alias = distances.data();
//
//  auto* previous_frontier = new char[num_nodes]; // AKA x-vector
//  std::fill(previous_frontier, previous_frontier + num_nodes, static_cast<char>(0));
//  previous_frontier[source] = 1;
//  auto* current_frontier = new char[num_nodes]; // AKA y-vector
//  std::fill(current_frontier, current_frontier + num_nodes, static_cast<char>(0));
//  auto* mask = new char[num_nodes]; // AKA t-vector
//  auto* parents = new char[num_nodes]; // AKA p-vector
//
//  std::size_t start = 0;
//  std::size_t end = 1;
//  std::size_t z = end;
//
//  int frontier_size = 0;
//  int distance = 1;
//  //#pragma acc kernel copy(distances_alias[:num_nodes]) copyin(row_indices[:row_indices_size], column_indices[:column_indices_size] previous_frontier[:num_nodes]) present(current_frontier_alias[:num_nodes])
//    do {
//      #pragma acc parallel for copyin(previous_frontier[:num_nodes])
//      for (std::size_t f = 0; f < num_nodes; f++) {
//      }
//      #pragma acc parallel for reduce(+, frontier_size) copyin(previous_frontier[:num_nodes])
//      for (std::size_t f = 0; f < num_nodes; f++) {
//        frontier_size += static_cast<int>(previous_frontier[f] >= 0);
//      }
//      distance += 1;
//    } while (frontier_size > 0);
//
//  delete[] previous_frontier;
//  delete[] current_frontier;
  return distances;
}