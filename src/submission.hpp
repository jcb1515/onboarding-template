#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// Views provide access to Grid storage without allocating or copying elements.
struct ConstGridView {
  const double* values;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  const double* row_data(const std::size_t row) const noexcept {
    return values + row * stride;
  }

  double operator()(const std::size_t row, const std::size_t col) const noexcept {
    return row_data(row)[col];
  }
};

struct GridView {
  double* values;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;

  double* row_data(const std::size_t row) const noexcept {
    return values + row * stride;
  }

  double& operator()(const std::size_t row, const std::size_t col) const noexcept {
    return row_data(row)[col];
  }
};

// Grid owns one contiguous row-major allocation and its lifetime.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  // Keeping stride separate from cols allows the layout to support padding.
  std::size_t stride_;
  std::vector<double> values_;

  static std::size_t checked_element_count(const std::size_t rows,
                                           const std::size_t stride) {
    if (stride != 0 && rows > std::numeric_limits<std::size_t>::max() / stride) {
      throw std::length_error("Grid dimensions are too large for one allocation");
    }

    return rows * stride;
  }

public:
  Grid(const std::size_t rows, const std::size_t cols)
      : rows_{rows}, cols_{cols}, stride_{cols},
        values_(checked_element_count(rows, cols), 0.0) {
  }

  std::size_t rows() const noexcept {
    return rows_;
  }

  std::size_t cols() const noexcept {
    return cols_;
  }

  ConstGridView read_view() const noexcept {
    return ConstGridView{values_.data(), rows_, cols_, stride_};
  }

  GridView write_view() noexcept {
    return GridView{values_.data(), rows_, cols_, stride_};
  }

  double& operator()(const std::size_t row, const std::size_t col) noexcept {
    return values_[row * stride_ + col];
  }

  double operator()(const std::size_t row, const std::size_t col) const noexcept {
    return values_[row * stride_ + col];
  }
};

namespace uwhpc_detail {

inline void copy_boundaries(const ConstGridView old_grid, const GridView new_grid) {
  if (old_grid.rows == 0 || old_grid.cols == 0) {
    return;
  }

  for (std::size_t col{0}; col < old_grid.cols; ++col) {
    new_grid(0, col) = old_grid(0, col);

    if (old_grid.rows > 1) {
      new_grid(old_grid.rows - 1, col) = old_grid(old_grid.rows - 1, col);
    }
  }

  for (std::size_t row{1}; row + 1 < old_grid.rows; ++row) {
    new_grid(row, 0) = old_grid(row, 0);

    if (old_grid.cols > 1) {
      new_grid(row, old_grid.cols - 1) = old_grid(row, old_grid.cols - 1);
    }
  }
}

inline void update_interior(const ConstGridView old_grid, const GridView new_grid) {
  if (old_grid.rows < 3 || old_grid.cols < 3) {
    return;
  }

  // Each row writes to a separate output range, so static partitioning is safe.
  #pragma omp parallel for schedule(static)
  for (std::size_t row = 1; row < old_grid.rows - 1; ++row) {
    const double* const above{old_grid.row_data(row - 1)};
    const double* const current{old_grid.row_data(row)};
    const double* const below{old_grid.row_data(row + 1)};
    double* const output{new_grid.row_data(row)};

    // Adjacent columns use contiguous memory and have no loop-carried writes.
    #pragma omp simd
    for (std::size_t col = 1; col < old_grid.cols - 1; ++col) {
      output[col] = 0.5 * current[col] + 0.125 * (above[col] + below[col] +
                                                  current[col - 1] + current[col + 1]);
    }
  }
}

}  // namespace uwhpc_detail

inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  if (&old_grid == &new_grid) {
    throw std::invalid_argument(
        "apply_stencil requires separate input and output grids");
  }

  if (old_grid.rows() != new_grid.rows() || old_grid.cols() != new_grid.cols()) {
    throw std::invalid_argument(
        "Input and output grids must have identical dimensions");
  }

  const ConstGridView old_view{old_grid.read_view()};
  const GridView new_view{new_grid.write_view()};

  // Complete the serial boundary phase before parallel interior writes begin.
  uwhpc_detail::copy_boundaries(old_view, new_view);
  uwhpc_detail::update_interior(old_view, new_view);
}
