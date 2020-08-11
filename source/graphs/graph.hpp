//
// Created by Daniel on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_GRAPH_HPP
#define CS535_GRAPHSEARCH_GRAPH_HPP

#include <utility>

#include "sparse_matrix.hpp"

struct Graph {
  using index_type = BinarySquareSparseMatrix::index_type;

  /**
   * @return True if the bidirectional edge exists
   */
  [[nodiscard]] bool Connected(index_type a, index_type b) const;

  /**
   * @return True if edge doesnt already exist.
   */
  bool AddEdge(index_type a, index_type b);

  /**
   * @return True if edge previously existed.
   */
  bool RemoveEdge(index_type a, index_type b);

  inline std::size_t NumEdges() const {
    return matrix.column_indices.size() / 2;
  }

  inline std::size_t NumNodes() const {
    return matrix.Height();
  }

  inline float AverageDiameter() const {
    float average = 0.0f;
    for (std::size_t i = 0; i < matrix.Height(); i++) {
      average += (matrix.row_indices.at(i + 1) - matrix.row_indices.at(i));
    }
    return average / static_cast<float>(matrix.Height());
  }

  /**
   * @note It is more efficient when generating a graph to call resize last as fewer in place modifications are needed.
   * @return True if connections were lost when the graph was shrunk.
   */
  inline bool Resize(std::size_t size) {
    return matrix.Resize(size);
  }

  std::ostream& ConvertForDot(std::ostream& os, const std::vector<int>& distances = std::vector<int>());

  friend std::ostream& operator<<(std::ostream& os, const Graph& graph) {
    os << "Graph: " << graph.matrix;
    return os;
  }

  BinarySquareSparseMatrix matrix;
};

#endif //CS535_GRAPHSEARCH_GRAPH_HPP
