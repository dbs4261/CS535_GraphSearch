//
// Created by Daniel on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_GRAPH_HPP
#define CS535_GRAPHSEARCH_GRAPH_HPP

#include <utility>
#include <ostream>

#include "coloring.hpp"
#include "sparse_matrix.hpp"

struct Graph {
  using index_type = BinarySquareSparseMatrix::index_type;

  /**
   * @return True if the bidirectional edge exists
   */
  [[nodiscard]] bool Connected(index_type a, index_type b) const {
    return (matrix.Contains(a, b) and matrix.Contains(b, a)) or a == b;
  }

  /**
   * @return True if edge doesnt already exist.
   */
  bool AddEdge(index_type a, index_type b) {
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

  /**
   * @return True if edge previously existed.
   */
  bool RemoveEdge(index_type a, index_type b) {
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

  std::size_t NumEdges() const {
    return matrix.column_indices.size() / 2;
  }

  std::size_t NumNodes() const {
    return matrix.Height();
  }

  /**
   * @note It is more efficient when generating a graph to call resize last as fewer in place modifications are needed.
   * @return True if connections were lost when the graph was shrunk.
   */
  bool Resize(std::size_t size) {
    return matrix.Resize(size);
  }

  std::ostream& ConvertForDot(std::ostream& os, const std::vector<int>& distances = std::vector<int>()) {
    int max_distance = distances.size() != NumNodes() ? -1 : *std::max_element(distances.begin(), distances.end());
    os << "strict graph CSR_Graph {\n";
    if (max_distance >= 0) {
      os << "node [colorscheme=\"rdylblu" << std::min(10, std::max(3, max_distance)) << "\"]\n";
    }
    // Nodes
    for (std::size_t i = 0; i < NumNodes(); i++) {
      os << i << " [label=\"" << i << "\"";
      if (max_distance >= 0 and distances.at(i) >= 0) {
        os << " color=\"" << brewer_spectral.at(std::min(10, distances.at(i))) << "\"";
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

  friend std::ostream& operator<<(std::ostream& os, const Graph& graph) {
    os << "Graph: " << graph.matrix;
    return os;
  }

  BinarySquareSparseMatrix matrix;
};

#endif //CS535_GRAPHSEARCH_GRAPH_HPP
