// Ported from OpenBMB/VoxCPM's MiniCPM4 stack (src/voxcpm/modules/minicpm4;
// Apache-2.0, Copyright OpenBMB; see licenses/VoxCPM.txt). Rewritten in C++
// and HIP for Gufo, with fused q/k/v and gate/up projections.

#include "transformer.hpp"

#include <cmath>

#include "kernels/generator_kernels.hpp"

namespace gufo::models::voxcpm {

KvCache::KvCache(const TransformerConfig& config, std::size_t capacity)
    : capacity_(capacity),
      layer_stride_(config.kv_heads * capacity * config.head_dim) {
  keys_ = DeviceBuffer<float>(config.layers * layer_stride_);
  values_ = DeviceBuffer<float>(config.layers * layer_stride_);
}

std::vector<float> RopeTable(const RopeConfig& rope, std::size_t positions) {
  // MiniCPMLongRoPE: inv_freq = 1 / base^(arange(0, dim, 2) / dim),
  // freqs = outer(t, 1 / factors) * inv_freq, scaled cos and sin, all in
  // float32.
  const std::size_t half = rope.dim / 2;
  const std::vector<double>& factors = rope.factors();
  const auto scaling = static_cast<float>(rope.scaling());
  std::vector<float> inverse(half);
  std::vector<float> factor_inverse(half);
  for (std::size_t i = 0; i < half; ++i) {
    const float exponent =
        static_cast<float>(2 * i) / static_cast<float>(rope.dim);
    inverse[i] = 1.0F / std::pow(static_cast<float>(rope.theta), exponent);
    factor_inverse[i] = 1.0F / static_cast<float>(factors[i]);
  }
  std::vector<float> table(positions * rope.dim);
  for (std::size_t position = 0; position < positions; ++position) {
    for (std::size_t i = 0; i < half; ++i) {
      const float angle =
          static_cast<float>(position) * factor_inverse[i] * inverse[i];
      table[position * rope.dim + i] = std::cos(angle) * scaling;
      table[position * rope.dim + half + i] = std::sin(angle) * scaling;
    }
  }
  return table;
}

void Project(const Projection& projection, const Bf16* x, float* y,
             std::size_t rows, Bf16Gemm& gemm, hipStream_t stream) {
  gemm.Linear(x, projection.weight, y, rows, projection.in, projection.out,
              false);
  if (projection.bias != nullptr)
    kernels::AddBias(y, projection.bias, rows, projection.out, stream);
}

void ForwardStack(const StackWeights& stack, const float* rope,
                  const AttentionLayout& layout, KvCache& cache, float* hidden,
                  float* output, std::size_t rows, StackScratch& scratch,
                  Bf16Gemm& gemm, hipStream_t stream) {
  const TransformerConfig& config = stack.config;
  const std::size_t width = config.hidden;
  const std::size_t q_width = config.heads * config.head_dim;
  const std::size_t qkv_width =
      (config.heads + 2 * config.kv_heads) * config.head_dim;
  scratch.normed.Reserve(rows * width);
  scratch.qkv.Reserve(rows * qkv_width);
  scratch.q.Reserve(rows * q_width);
  scratch.attention.Reserve(rows * q_width);
  scratch.gate_up.Reserve(rows * 2 * config.intermediate);
  scratch.activation.Reserve(rows * config.intermediate);
  for (std::size_t index = 0; index < stack.layers.size(); ++index) {
    const LayerWeights& layer = stack.layers[index];
    kernels::RmsNorm(hidden, layer.input_norm, scratch.normed.get(), nullptr,
                     rows, width, config.rms_eps, stream);
    gemm.Linear(scratch.normed.get(), layer.qkv.weight, scratch.qkv.get(), rows,
                width, qkv_width, false);
    kernels::QkvRope(scratch.qkv.get(), config.rope ? rope : nullptr,
                     layout.positions, layout.cache_rows, scratch.q.get(),
                     cache.keys(index), cache.values(index), rows, config.heads,
                     config.kv_heads, config.head_dim, cache.capacity(),
                     stream);
    kernels::Attention(scratch.q.get(), cache.keys(index), cache.values(index),
                       scratch.attention.get(), rows, layout.rows_per_sequence,
                       layout.keys_per_sequence, layout.past, layout.causal,
                       config.heads, config.kv_heads, config.head_dim,
                       cache.capacity(), stream);
    gemm.Linear(scratch.attention.get(), layer.o.weight, hidden, rows, q_width,
                width, true);
    kernels::RmsNorm(hidden, layer.post_norm, scratch.normed.get(), nullptr,
                     rows, width, config.rms_eps, stream);
    gemm.Linear(scratch.normed.get(), layer.gate_up.weight,
                scratch.gate_up.get(), rows, width, 2 * config.intermediate,
                false);
    kernels::SwiGlu(scratch.gate_up.get(), scratch.activation.get(), rows,
                    config.intermediate, stream);
    gemm.Linear(scratch.activation.get(), layer.down.weight, hidden, rows,
                config.intermediate, width, true);
  }
  kernels::RmsNorm(hidden, stack.norm, scratch.normed.get(), output, rows,
                   width, config.rms_eps, stream);
}

}  // namespace gufo::models::voxcpm
