#ifndef GUFO_MODELS_VOXCPM_MATRIX_HPP_
#define GUFO_MODELS_VOXCPM_MATRIX_HPP_

#include <cstdint>

namespace gufo::models::voxcpm {

/// How a weight matrix is stored on the device.
enum class WeightFormat : std::uint8_t {
  kBf16,       // bfloat16 [out][in], row-major
  kBf16Tiles,  // bfloat16 tiles for the matrix cores (wmma_products.hpp)
  kQ8Tiles,    // GGML Q8_0 tiles for the matrix cores (wmma_products.hpp)
};

/// A weight matrix [out][in] on the device.
struct Matrix {
  WeightFormat format = WeightFormat::kBf16;
  const void* data = nullptr;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_MATRIX_HPP_
