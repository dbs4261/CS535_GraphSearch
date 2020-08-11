//
// Created by Daniel Simon on 8/5/20.
//

#include "graph.hpp"

#include <algorithm>
#include <cassert>
#include <ostream>

#include "coloring.hpp"

[[nodiscard]] bool Graph::Connected(Graph::index_type a, Graph::index_type b) const {
  return (matrix.Contains(a, b) and matrix.Contains(b, a)) or a == b;
}

bool Graph::AddEdge(Graph::index_type a, Graph::index_type b) {
  if (Connected(a, b)) {
    return false;
  } else {
    bool forward = matrix.Set(a, b, true);
    bool backward = matrix.Set(b, a, true);
    assert(not (forward xor backward));
    // Set will return false when the added edge is new
    return not forward and not backward;
  }
}

bool Graph::RemoveEdge(Graph::index_type a, Graph::index_type b) {
  if (a == b) {
    return false;
  } else {
    // Set will return false when the removed edge previously existed
    bool forward = matrix.Set(a, b, false);
    bool backward = matrix.Set(b, a, false);
    assert(not(forward xor backward));
    return not forward and not backward;
  }
}

std::ostream& Graph::ConvertForDot(std::ostream& os, const std::vector<int>& distances) {
  int max_distance = distances.size() != NumNodes() ? -1 : *std::max_element(distances.begin(), distances.end());
  os << "strict graph CSR_Graph {\n";
  // Nodes
  for (std::size_t i = 0; i < NumNodes(); i++) {
    os << i << " [label=\"" << i << "\"";
    if (max_distance >= 0 and distances.at(i) >= 0) {
      os << " color=\"" << brewer_spectral.at(std::min<int>(brewer_spectral.size() - 1, distances.at(i))) << "\"";
    } else {
      os << "color=\"black\"";
    }
    os << "]\n";
  }
  // Edges
  for (std::size_t i = 0; i < NumNodes(); i++) {
    auto cols_start = matrix.row_indices.at(i);
    auto cols_end = matrix.row_indices.at(i + 1);
    if (cols_end - cols_start > 0) {
      os << i << " -- {";
      for (std::size_t j = cols_start; j < cols_end; j++) {
        os << matrix.column_indices.at(j) << " ";
      }
      os << "}\n";
    }
  }
  os << "}\n";
  return os;
}