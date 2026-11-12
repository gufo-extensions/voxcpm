#ifndef GUFO_MODELS_VOXCPM_WEIGHTS_HPP_
#define GUFO_MODELS_VOXCPM_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "device.hpp"
#include "matrix.hpp"
#include "model_files.hpp"

namespace gufo::models::voxcpm {

/// Raw bfloat16 bits.
using Bf16 = std::uint16_t;

/// A linear layer y = x W^T (+ b).
struct Projection {
  Matrix weight;
  const Bf16* bias = nullptr;
  std::size_t in = 0;
  std::size_t out = 0;
};

struct LayerWeights {
  const Bf16* input_norm = nullptr;
  Projection qkv;  // [q heads | k heads | v heads]
  Projection o;
  const Bf16* post_norm = nullptr;
  Projection gate_up;  // [gate | up]
  Projection down;
};

struct StackWeights {
  TransformerConfig config;
  std::vector<LayerWeights> layers;
  const Bf16* norm = nullptr;
};

/// The generator's weights on the device, with q/k/v and gate/up of every
/// layer placed side by side so that each pair or triple is one product.
/// GGML Q8_0 matrices stay quantized, in tiles; the DiT's bfloat16
/// matrices are tiled too (its products have 22 rows); everything else is
/// bfloat16 rows.
struct GeneratorWeights {
  StackWeights base_lm;
  StackWeights residual_lm;
  StackWeights encoder;
  StackWeights dit;
  const Bf16* embed = nullptr;    // [vocab][hidden]
  const Bf16* special = nullptr;  // local encoder's CLS row
  Projection encoder_in;          // feat_encoder.in_proj
  Projection enc_to_lm;
  Projection lm_to_dit;
  Projection res_to_dit;
  Projection fusion;
  Projection fsq_in;
  Projection fsq_out;
  Projection stop_proj;
  Projection stop_head;
  Projection dit_in;
  Projection dit_cond;
  Projection dit_out;
  Projection time_1;
  Projection time_2;
  Projection delta_time_1;
  Projection delta_time_2;
};

/// Owns the device copy of the generator's weights.
class DeviceWeights {
public:
  /// Checks every tensor's shape against `config` and uploads: Q8_0
  /// matrices as Q8_0 tiles; other GGUF types (F16, F32) and Q8_0 vectors
  /// and embedding tables converted to bfloat16.
  [[nodiscard]] static std::unique_ptr<DeviceWeights> Upload(
      const ModelFiles& files, const Config& config,
      std::string* error = nullptr);

  [[nodiscard]] const GeneratorWeights& weights() const { return weights_; }
  [[nodiscard]] std::size_t bytes() const { return storage_.size(); }

private:
  DeviceWeights() = default;

  DeviceBuffer<std::uint8_t> storage_;
  GeneratorWeights weights_;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_WEIGHTS_HPP_
