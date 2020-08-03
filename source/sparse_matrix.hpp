//
// Created by Daniel Simon on 8/2/20.
//

#ifndef CS535_GRAPHSEARCH_SPARSE_MATRIX_HPP
#define CS535_GRAPHSEARCH_SPARSE_MATRIX_HPP

#include <algorithm>
#include <vector>
#include <ostream>

struct BinarySquareSparseMatrix {
  using index_type = std::size_t;
  BinarySquareSparseMatrix() : column_indices(), row_indices(2, 0) {}

  [[nodiscard]] bool Contains(index_type x, index_type y) const {
    if (y >= Height()) {
      return false;
    }
    if (row_indices.at(y + 1) - row_indices.at(y) > 0) {
      return std::any_of(column_indices.begin() + row_indices.at(y),
                         column_indices.begin() + row_indices.at(y + 1),
                         [x](index_type column){return column == x;});
    } else {
      return false;
    }
  }

  /**
   * @param x The column the value is in.
   * @param y The row the value is in.
   * @param state
   * @return True if the value == state
   */
  [[maybe_unused]] bool Set(index_type x, index_type y, bool state) {
    if (state) {
      // Add value
      if (y > Height() - 1) {
        // Grow matrix and emplace
        column_indices.emplace_back(x);
        index_type last_index = row_indices.back();
        std::fill_n(std::back_inserter(row_indices), y - Height(), last_index);
        row_indices.emplace_back(last_index + 1);
        return false;
      } else {
        auto cols_start = column_indices.begin() + row_indices.at(y);
        auto cols_end = column_indices.begin() + row_indices.at(y + 1);
        if (std::distance(cols_start, cols_end) != 0) {
          auto it = std::lower_bound(cols_start, cols_end, x);
          if (*it == x and it != cols_end) {
            // Value already exists
            return true;
          } else {
            column_indices.insert(it, x);
          }
        } else {
          column_indices.insert(cols_start, x);
        }
        // Value doesnt exist
        std::for_each(row_indices.begin() + y + 1, row_indices.end(), [](index_type& val){return val += 1;});
        return false;
      }
    } else {
      // Remove value
      if (y > Height()) {
        // Cant remove if out of bounds
        return true;
      } else {
        auto it = std::find(column_indices.begin() + row_indices.at(y),
                            column_indices.begin() + row_indices.at(y + 1), x);
        if (it == column_indices.begin() + row_indices.at(y + 1) and
            row_indices.at(y) != row_indices.at(y + 1)) {
          // Value isnt set
          return true;
        } else {
          column_indices.erase(it);
          std::for_each(row_indices.begin() + y + 1, row_indices.end(), [](index_type& val){val -= 1;});
          return false;
        }
      }
    }
  }

  /**
   * @return The EFFECTIVE width, empty columns arent actually stored so the true width is undefined.
   */
  [[nodiscard]] index_type Width() const {
    return *std::max_element(column_indices.begin(), column_indices.end()) + 1;
  }

  [[nodiscard]] index_type Height() const {
    return row_indices.size() - 1;
  }

  /**
   * @return True data was removed when the data was shrunk
   */
  bool Resize(std::size_t size) {
    if (size == Height()) {
      return false;
    } else if (size < Height()) {
      index_type max_row_index = row_indices.at(size + 1);
      bool out = max_row_index == row_indices.back();
      column_indices.erase(column_indices.begin() + max_row_index, column_indices.end());
      row_indices.erase(row_indices.begin() + size + 1, row_indices.end());
      // First change the row indices
      for (std::size_t i = 0; i < row_indices.size() - 1; i++) {
        auto count = std::count_if(column_indices.begin() + row_indices.at(i), column_indices.begin() + row_indices.at(i + 1),
            [size](index_type val){
          return val >= size;
        });
        std::for_each(row_indices.begin() + i, row_indices.end(), [count](index_type& val){val -= count;});
      }
      // Then remove the columns
      auto remove_it = std::remove_if(column_indices.begin(), column_indices.end(),
          [size](index_type val){
        return val >= size;
      });
      // If column data was remove report it
      out or_eq remove_it == column_indices.end();
      column_indices.erase(remove_it, column_indices.end());
      return out;
    } else {
      index_type last_index = row_indices.back();
      std::fill_n(std::back_inserter(row_indices), size - Height(), last_index);
      return false;
    }
  }


  friend std::ostream& operator<<(std::ostream& os, const BinarySquareSparseMatrix& matrix) {
    os << "Sparse Matrix: " << matrix.Width() << ", " << matrix.Height() << "\n";
    for (std::size_t i = 0; i < matrix.row_indices.size() - 1; i++) {
      index_type previous = 0;
      for (std::size_t j = matrix.row_indices.at(i); j < matrix.row_indices.at(i + 1); j++) {
        for (std::size_t k = previous; k < matrix.column_indices.at(j); k++) {
          os << "0 ";
        }
        os << "1 ";
        previous = matrix.column_indices.at(j) + 1;
      }
      for (std::size_t k = previous; k < matrix.Height(); k++) {
        os << "0 ";
      }
      os << "\n";
    }
    return os;
  }

  std::vector<index_type> column_indices;
  std::vector<index_type> row_indices;
};

#endif //CS535_GRAPHSEARCH_SPARSE_MATRIX_HPP
