#ifndef GUFO_MODELS_VOXCPM_TRANSFORMER_HPP_
#define GUFO_MODELS_VOXCPM_TRANSFORMER_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.hpp"
#include "device.hpp"
#include "gemm.hpp"
#include "weights.hpp"

namespace gufo::models::voxcpm {

/// Keys and values of one stack: [layer][kv_head][capacity][head_dim] each.
class KvCache {
public:
  KvCache() = default;
  KvCache(const TransformerConfig& config, std::size_t capacity);

  [[nodiscard]] std::size_t capacity() const { return capacity_; }
  [[nodiscard]] float* keys(std::size_t layer) const {
    return keys_.get() + layer * layer_stride_;
  }
  [[nodiscard]] float* values(std::size_t layer) const {
    return values_.get() + layer * layer_stride_;
  }

private:
  DeviceBuffer<float> keys_;
  DeviceBuffer<float> values_;
  std::size_t capacity_ = 0;
  std::size_t layer_stride_ = 0;
};

/// Which keys each row attends to; see kernels::Attention. `positions`
/// (rotary table rows) and `cache_rows` are device arrays of one entry per
/// row.
struct AttentionLayout {
  const std::int32_t* positions = nullptr;
  const std::int32_t* cache_rows = nullptr;
  std::size_t rows_per_sequence = 0;
  std::size_t keys_per_sequence = 0;
  std::size_t past = 0;
  bool causal = true;
};

/// Scratch shared by the stacks of one generator.
struct StackScratch {
  DeviceBuffer<Bf16> normed;
  DeviceBuffer<float> qkv;
  DeviceBuffer<float> q;
  DeviceBuffer<Bf16> attention;
  DeviceBuffer<float> gate_up;
  DeviceBuffer<Bf16> activation;
};

/// The rotary table [position][cos (head_dim / 2) | sin (head_dim / 2)] of
/// MiniCPM's LongRoPE, computed in float32 as the reference does.
[[nodiscard]] std::vector<float> RopeTable(const RopeConfig& rope,
                                           std::size_t positions);

/// Runs one MiniCPM4 stack over `rows` rows of the float32 residual stream
/// `hidden` (updated in place) and writes the final norm's output to
/// `output` (float32). `rope` is the device rotary table, null for stacks
/// without rotary embedding.
void ForwardStack(const StackWeights& stack, const float* rope,
                  const AttentionLayout& layout, KvCache& cache, float* hidden,
                  float* output, std::size_t rows, StackScratch& scratch,
                  Bf16Gemm& gemm, hipStream_t stream);

/// y[rows][out] = x[rows][in] W^T + b with bf16 x.
void Project(const Projection& projection, const Bf16* x, float* y,
             std::size_t rows, Bf16Gemm& gemm, hipStream_t stream);

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_TRANSFORMER_HPP_
