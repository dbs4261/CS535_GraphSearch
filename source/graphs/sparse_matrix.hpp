//
// Created by Daniel Simon on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_SPARSE_MATRIX_HPP
#define CS535_GRAPHSEARCH_SPARSE_MATRIX_HPP

#include <vector>
#include <ostream>

struct BinarySquareSparseMatrix {
  using index_type = std::size_t;
  BinarySquareSparseMatrix() : column_indices(), row_indices(2, 0) {}

  [[nodiscard]] bool Contains(index_type x, index_type y) const;

  /**
   * @param x The column the value is in.
   * @param y The row the value is in.
   * @param state
   * @return True if the value == state
   */
  [[maybe_unused]] bool Set(index_type x, index_type y, bool state);

  /**
   * @return The EFFECTIVE width, empty columns arent actually stored so the true width is undefined.
   */
  [[nodiscard]] index_type Width() const;

  [[nodiscard]] index_type Height() const;

  /**
   * @return True data was removed when the data was shrunk
   */
  bool Resize(std::size_t size);

  friend std::ostream& operator<<(std::ostream& os, const BinarySquareSparseMatrix& matrix);

  std::vector<index_type> column_indices;
  std::vector<index_type> row_indices;
};

#endif //CS535_GRAPHSEARCH_SPARSE_MATRIX_HPP
