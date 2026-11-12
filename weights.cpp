#include "weights.hpp"

#include <hip/hip_runtime.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/convert_kernels.hpp"
#include "kernels/wmma_products.hpp"

namespace gufo::models::voxcpm {
namespace {

class WeightError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

constexpr std::size_t kAlignment = 16;

/// Copies tensors one after another into the device buffer: Q8_0 matrices
/// as Q8_0 tiles, other matrices as bfloat16 rows or tiles, vectors as
/// bfloat16. Without a buffer (`base` null) it only counts the bytes.
class Uploader {
public:
  Uploader(const ModelFiles& files, std::uint8_t* base, std::size_t capacity)
      : files_(files), base_(base), capacity_(capacity) {}

  /// A vector (or the embedding table) as bfloat16.
  const Bf16* Take(const std::string& name,
                   std::initializer_list<std::size_t> shape) {
    const Tensor& tensor = Find(name, shape);
    const std::size_t elements = tensor.Elements();
    auto* target = static_cast<Bf16*>(Reserve(elements * sizeof(Bf16)));
    if (target != nullptr)
      ToBf16(tensor, name, target);
    return target;
  }

  /// A [out][in] matrix: Q8_0 tiles for a Q8_0 tensor when `q8`, bfloat16
  /// tiles when `tiles`, bfloat16 rows otherwise or when the columns don't
  /// fill whole tiles.
  Matrix TakeMatrix(const std::string& name, std::size_t out, std::size_t in,
                    bool tiles, bool q8) {
    const Tensor& tensor = Find(name, {out, in});
    Matrix matrix;
    matrix.format = Format(tensor.dtype == DType::kQ8_0 && q8, tiles, in);
    void* target = Reserve(Bytes(matrix.format, out, in));
    matrix.data = target;
    if (target != nullptr)
      Write(tensor, name, matrix.format, out, in, target);
    return matrix;
  }

  /// `names` ([outs[i]][in] each) side by side as one matrix: Q8_0 tiles
  /// when all are Q8_0, bfloat16 tiles when `tiles`, bfloat16 rows
  /// otherwise or when a part doesn't fill whole tiles of rows.
  Matrix Fused(const std::vector<std::string>& names,
               const std::vector<std::size_t>& outs, std::size_t in,
               bool tiles) {
    std::vector<const Tensor*> tensors;
    bool q8 = true;
    bool whole_tiles = true;
    for (std::size_t index = 0; index < names.size(); ++index) {
      tensors.push_back(&Find(names[index], {outs[index], in}));
      q8 = q8 && tensors.back()->dtype == DType::kQ8_0;
      whole_tiles = whole_tiles && outs[index] % kernels::kTileRows == 0;
    }
    Matrix matrix;
    matrix.format = Format(q8 && whole_tiles, tiles && whole_tiles, in);
    std::size_t total = 0;
    for (const std::size_t out : outs)
      total += Bytes(matrix.format, out, in);
    auto* target = static_cast<std::uint8_t*>(Reserve(total));
    matrix.data = target;
    if (target == nullptr)
      return matrix;
    for (std::size_t index = 0; index < names.size(); ++index) {
      Write(*tensors[index], names[index], matrix.format, outs[index], in,
            target);
      target += Bytes(matrix.format, outs[index], in);
    }
    return matrix;
  }

  Projection Linear(const std::string& prefix, std::size_t in, std::size_t out,
                    bool bias, bool tiles = false) {
    Projection projection;
    projection.weight = TakeMatrix(prefix + ".weight", out, in, tiles, true);
    if (bias)
      projection.bias = Take(prefix + ".bias", {out});
    projection.in = in;
    projection.out = out;
    return projection;
  }

  /// One MiniCPM4 stack; `tiles` stores its bfloat16 matrices as tiles.
  StackWeights Stack(const std::string& prefix, const TransformerConfig& config,
                     bool tiles) {
    StackWeights stack;
    stack.config = config;
    const std::size_t hidden = config.hidden;
    const std::size_t q = config.heads * config.head_dim;
    const std::size_t kv = config.kv_heads * config.head_dim;
    for (std::size_t index = 0; index < config.layers; ++index) {
      const std::string layer = prefix + ".layers." + std::to_string(index);
      LayerWeights weights;
      weights.input_norm = Take(layer + ".input_layernorm.weight", {hidden});
      const std::string attention = layer + ".self_attn.";
      weights.qkv.weight =
          Fused({attention + "q_proj.weight", attention + "k_proj.weight",
                 attention + "v_proj.weight"},
                {q, kv, kv}, hidden, tiles);
      weights.qkv.in = hidden;
      weights.qkv.out = q + 2 * kv;
      weights.o = Linear(attention + "o_proj", q, hidden, false, tiles);
      weights.post_norm =
          Take(layer + ".post_attention_layernorm.weight", {hidden});
      weights.gate_up.weight = Fused(
          {layer + ".mlp.gate_proj.weight", layer + ".mlp.up_proj.weight"},
          {config.intermediate, config.intermediate}, hidden, tiles);
      weights.gate_up.in = hidden;
      weights.gate_up.out = 2 * config.intermediate;
      weights.down = Linear(layer + ".mlp.down_proj", config.intermediate,
                            hidden, false, tiles);
      stack.layers.push_back(weights);
    }
    stack.norm = Take(prefix + ".norm.weight", {hidden});
    return stack;
  }

  [[nodiscard]] std::size_t used() const { return used_; }
  [[nodiscard]] std::size_t taken() const { return taken_; }

private:
  static WeightFormat Format(bool q8, bool tiles, std::size_t in) {
    if (in % kernels::kTileColumns != 0)
      return WeightFormat::kBf16;
    if (q8)
      return WeightFormat::kQ8Tiles;
    return tiles ? WeightFormat::kBf16Tiles : WeightFormat::kBf16;
  }

  static std::size_t Bytes(WeightFormat format, std::size_t out,
                           std::size_t in) {
    switch (format) {
      case WeightFormat::kQ8Tiles:
        return kernels::Q8TilesBytes(out, in);
      case WeightFormat::kBf16Tiles:
        return kernels::Bf16TilesBytes(out, in);
      case WeightFormat::kBf16:
        break;
    }
    return out * in * sizeof(Bf16);
  }

  /// Writes `tensor` in `format` at `target`.
  void Write(const Tensor& tensor, const std::string& name, WeightFormat format,
             std::size_t out, std::size_t in, void* target) {
    switch (format) {
      case WeightFormat::kBf16:
        ToBf16(tensor, name, static_cast<Bf16*>(target));
        return;
      case WeightFormat::kBf16Tiles:
        rows_.Reserve(out * in);
        ToBf16(tensor, name, rows_.get());
        Check(hipMemset(target, 0, Bytes(format, out, in)),
              "hipMemset weights");
        kernels::Bf16ToTiles(rows_.get(), out, in, static_cast<Bf16*>(target),
                             nullptr);
        Check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        return;
      case WeightFormat::kQ8Tiles:
        CheckQ8(tensor, name);
        Stage(tensor);
        Check(hipMemset(target, 0, Bytes(format, out, in)),
              "hipMemset weights");
        kernels::Q8_0ToTiles(
            reinterpret_cast<const std::uint8_t*>(staging_.get()), out, in,
            static_cast<std::uint8_t*>(target), nullptr);
        Check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        return;
    }
  }

  const Tensor& Find(const std::string& name,
                     std::initializer_list<std::size_t> shape) {
    const Tensor* tensor = files_.Model(name);
    if (tensor == nullptr)
      throw WeightError("the checkpoint lacks " + name);
    if (!SameShape(tensor->shape, std::vector<std::size_t>(shape)))
      throw WeightError(name + " has an unexpected shape");
    ++taken_;
    return *tensor;
  }

  /// The next `bytes` of the buffer (null when counting).
  void* Reserve(std::size_t bytes) {
    used_ = (used_ + kAlignment - 1) / kAlignment * kAlignment;
    void* target = nullptr;
    if (base_ != nullptr) {
      if (used_ + bytes > capacity_)
        throw WeightError("weights exceed the planned device buffer");
      target = base_ + used_;
    }
    used_ += bytes;
    return target;
  }

  void Stage(const Tensor& tensor) {
    staging_.Reserve((tensor.bytes + 3) / 4);
    Check(hipMemcpy(staging_.get(), tensor.data, tensor.bytes,
                    hipMemcpyHostToDevice),
          "hipMemcpy weights");
  }

  static void CheckQ8(const Tensor& tensor, const std::string& name) {
    const std::size_t elements = tensor.Elements();
    if (elements % 32 != 0 || tensor.bytes != elements / 32 * 34)
      throw WeightError(name + " is not whole Q8_0 blocks");
  }

  void ToBf16(const Tensor& tensor, const std::string& name, Bf16* target) {
    const std::size_t elements = tensor.Elements();
    switch (tensor.dtype) {
      case DType::kBF16:
        Check(
            hipMemcpy(target, tensor.data, tensor.bytes, hipMemcpyHostToDevice),
            "hipMemcpy weights");
        return;
      case DType::kF16:
      case DType::kF32:
      case DType::kQ8_0: {
        if (tensor.dtype == DType::kQ8_0)
          CheckQ8(tensor, name);
        Stage(tensor);
        const void* raw = staging_.get();
        if (tensor.dtype == DType::kF16)
          kernels::HalfToBf16(static_cast<const std::uint16_t*>(raw), target,
                              elements, nullptr);
        else if (tensor.dtype == DType::kF32)
          kernels::FloatToBf16(static_cast<const float*>(raw), target, elements,
                               nullptr);
        else
          kernels::Q8_0ToBf16(static_cast<const std::uint8_t*>(raw), target,
                              elements, nullptr);
        Check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        return;
      }
      default:
        throw WeightError(name + " has an unsupported type " +
                          std::string(DTypeName(tensor.dtype)));
    }
  }

  const ModelFiles& files_;
  DeviceBuffer<std::uint32_t> staging_;
  DeviceBuffer<Bf16> rows_;
  std::uint8_t* base_;
  std::size_t capacity_;
  std::size_t used_ = 0;
  std::size_t taken_ = 0;
};

/// Takes every tensor of the generator in its order.
GeneratorWeights TakeAll(Uploader& take, const Config& config) {
  GeneratorWeights w;
  const std::size_t hidden = config.lm.hidden;
  const std::size_t local = config.encoder.hidden;
  const std::size_t dit = config.dit.hidden;
  const std::size_t feat = config.feat_dim;

  w.embed = take.Take("base_lm.embed_tokens.weight", {config.lm.vocab, hidden});
  w.base_lm = take.Stack("base_lm", config.lm, false);
  w.residual_lm = take.Stack("residual_lm", config.residual_lm, false);
  w.encoder = take.Stack("feat_encoder.encoder", config.encoder, false);
  w.special = take.Take("feat_encoder.special_token", {1, 1, 1, local});
  w.encoder_in = take.Linear("feat_encoder.in_proj", feat, local, true);
  // The DiT's products have 22 rows (the CFG pair of a patch): tiles.
  w.dit = take.Stack("feat_decoder.estimator.decoder", config.dit, true);
  const std::string estimator = "feat_decoder.estimator.";
  w.dit_in = take.Linear(estimator + "in_proj", feat, dit, true);
  w.dit_cond = take.Linear(estimator + "cond_proj", feat, dit, true);
  w.dit_out = take.Linear(estimator + "out_proj", dit, feat, true);
  w.time_1 = take.Linear(estimator + "time_mlp.linear_1", dit, dit, true);
  w.time_2 = take.Linear(estimator + "time_mlp.linear_2", dit, dit, true);
  w.delta_time_1 =
      take.Linear(estimator + "delta_time_mlp.linear_1", dit, dit, true);
  w.delta_time_2 =
      take.Linear(estimator + "delta_time_mlp.linear_2", dit, dit, true);
  w.enc_to_lm = take.Linear("enc_to_lm_proj", local, hidden, true);
  w.lm_to_dit = take.Linear("lm_to_dit_proj", hidden, dit, true);
  w.res_to_dit = take.Linear("res_to_dit_proj", hidden, dit, true);
  w.fusion = take.Linear("fusion_concat_proj", 2 * hidden, hidden, true);
  w.fsq_in =
      take.Linear("fsq_layer.in_proj", hidden, config.fsq_latent_dim, true);
  w.fsq_out =
      take.Linear("fsq_layer.out_proj", config.fsq_latent_dim, hidden, true);
  w.stop_proj = take.Linear("stop_proj", hidden, hidden, true);
  w.stop_head = take.Linear("stop_head", hidden, 2, false);
  return w;
}

}  // namespace

std::unique_ptr<DeviceWeights> DeviceWeights::Upload(const ModelFiles& files,
                                                     const Config& config,
                                                     std::string* error) {
  std::unique_ptr<DeviceWeights> device(new DeviceWeights());
  try {
    // Count the bytes of the chosen formats first, then upload.
    Uploader count(files, nullptr, 0);
    (void)TakeAll(count, config);
    if (count.taken() != files.model_tensors().size())
      throw WeightError("the checkpoint holds tensors VoxCPM2 does not use");
    device->storage_ = DeviceBuffer<std::uint8_t>(count.used());
    Uploader take(files, device->storage_.get(), count.used());
    device->weights_ = TakeAll(take, config);
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return nullptr;
  }
  return device;
}

}  // namespace gufo::models::voxcpm
