// Ported from OpenBMB/VoxCPM's AudioVAE V2
// (src/voxcpm/modules/audiovae/audio_vae_v2.py; Apache-2.0, Copyright OpenBMB;
// see licenses/VoxCPM.txt), whose convolution blocks and Snake activation
// follow Descript's DAC (MIT, Copyright Descript; see licenses/DAC.txt).
// Rewritten in C++ and HIP for Gufo: time-major float32, streaming
// convolution state, weight norm folded at load.

#include "vae.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels/vae_kernels.hpp"
#include "torch_checkpoint.hpp"

namespace gufo::models::voxcpm {
namespace {

class VaeError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

constexpr std::size_t kKernel = 7;  // residual units and the outer convs
constexpr std::array<std::size_t, 3> kDilations = {1, 3, 9};

/// Offsets into the packed float32 weights.
using Offset = std::size_t;

struct ResidualUnit {
  std::size_t dilation = 1;
  Offset alpha_in = 0;
  Offset depthwise = 0;  // [channels][7]
  Offset depthwise_bias = 0;
  Offset alpha_out = 0;
  Offset pointwise = 0;  // [channels][channels]
  Offset pointwise_bias = 0;
};

struct DecoderBlock {
  std::size_t in = 0;
  std::size_t out = 0;
  std::size_t stride = 0;
  Offset scale = 0;  // rate conditioning of the block's input
  Offset bias = 0;
  Offset alpha = 0;
  // ConvTranspose as two products: out[t][(r, c)] =
  // A[(r, c)] x[t] + B[(r, c)] x[t - 1] + bias[c].
  Offset up_current = 0;   // [stride * out][in]
  Offset up_previous = 0;  // [stride * out][in]
  Offset up_bias = 0;      // [stride * out]
  std::array<ResidualUnit, 3> units;
};

struct EncoderBlock {
  std::size_t in = 0;
  std::size_t out = 0;
  std::size_t stride = 0;
  std::array<ResidualUnit, 3> units;
  Offset alpha = 0;
  Offset down = 0;  // [2 * stride][out][in], one matrix per tap
  Offset down_bias = 0;
};

}  // namespace

struct AudioVae::Weights {
  Offset in_depthwise = 0;  // [latent][7]
  Offset in_depthwise_bias = 0;
  Offset in_pointwise = 0;  // [decoder_dim][latent]
  Offset in_pointwise_bias = 0;
  std::vector<DecoderBlock> blocks;
  Offset out_alpha = 0;
  Offset out_conv = 0;  // [channels][7]
  float out_bias = 0.0F;
  std::size_t out_channels = 0;

  Offset encoder_in = 0;  // [encoder_dim][7]
  Offset encoder_in_bias = 0;
  std::vector<EncoderBlock> encoder_blocks;
  Offset mu = 0;  // [3][latent][channels]
  Offset mu_bias = 0;
  std::size_t mu_channels = 0;
};

namespace {

float FromBf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

float FromHalf(std::uint16_t bits) {
  const std::uint32_t sign = (bits & 0x8000U) << 16U;
  std::uint32_t exponent = (bits >> 10U) & 0x1FU;
  std::uint32_t mantissa = bits & 0x3FFU;
  std::uint32_t wide = 0;
  if (exponent == 0x1FU) {
    wide = sign | 0x7F800000U | (mantissa << 13U);
  } else if (exponent != 0) {
    wide = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  } else if (mantissa != 0) {
    exponent = 113;
    while ((mantissa & 0x400U) == 0) {
      mantissa <<= 1U;
      --exponent;
    }
    wide = sign | (exponent << 23U) | ((mantissa & 0x3FFU) << 13U);
  } else {
    wide = sign;
  }
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

/// Reads the checkpoint, folds weight normalization and packs the float32
/// weights in the layouts the products use.
class Packer {
public:
  explicit Packer(const ModelFiles& files) : files_(files) {}

  std::vector<float>& data() { return data_; }

  /// The tensor `name` as float32 (stored as F32, F16 or BF16).
  std::vector<float> Get(const std::string& name,
                         const std::vector<std::size_t>& shape) const {
    const Tensor* tensor = files_.Vae(name);
    if (tensor == nullptr)
      throw VaeError("the AudioVAE weights lack " + name);
    if (!SameShape(tensor->shape, shape))
      throw VaeError("unexpected shape of the AudioVAE tensor " + name);
    const std::size_t count = tensor->Elements();
    std::vector<float> values(count);
    if (tensor->dtype == DType::kF32) {
      std::memcpy(values.data(), tensor->data, count * sizeof(float));
      return values;
    }
    if (tensor->dtype != DType::kF16 && tensor->dtype != DType::kBF16)
      throw VaeError("unsupported type of the AudioVAE tensor " + name);
    for (std::size_t i = 0; i < count; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, tensor->data + 2 * i, sizeof(bits));
      values[i] =
          tensor->dtype == DType::kBF16 ? FromBf16(bits) : FromHalf(bits);
    }
    return values;
  }

  Offset Raw(const std::string& name, const std::vector<std::size_t>& shape) {
    return Append(Get(name, shape));
  }

  /// weight = g * v / |v|, the norm over all but dimension 0 (torch
  /// weight_norm with dim=0), in the checkpoint's layout.
  std::vector<float> Normalized(const std::string& prefix,
                                const std::vector<std::size_t>& shape) {
    const std::vector<float> v = Get(prefix + ".weight_v", shape);
    const std::vector<float> g = Get(prefix + ".weight_g", {shape[0], 1, 1});
    const std::size_t rows = shape[0];
    const std::size_t width = v.size() / rows;
    std::vector<float> weight(v.size());
    for (std::size_t row = 0; row < rows; ++row) {
      double norm = 0.0;
      for (std::size_t i = 0; i < width; ++i) {
        const double value = v[row * width + i];
        norm += value * value;
      }
      const double factor = g[row] / std::sqrt(norm);
      for (std::size_t i = 0; i < width; ++i)
        weight[row * width + i] =
            static_cast<float>(v[row * width + i] * factor);
    }
    return weight;
  }

  Offset Append(const std::vector<float>& values) {
    const Offset offset = data_.size();
    data_.insert(data_.end(), values.begin(), values.end());
    return offset;
  }

  /// A Conv1d [out][in][k] as one [out][in] matrix per tap: [k][out][in].
  Offset PerTap(const std::string& prefix, std::size_t out, std::size_t in,
                std::size_t kernel) {
    const std::vector<float> w = Normalized(prefix, {out, in, kernel});
    std::vector<float> packed(w.size());
    for (std::size_t o = 0; o < out; ++o)
      for (std::size_t i = 0; i < in; ++i)
        for (std::size_t k = 0; k < kernel; ++k)
          packed[(k * out + o) * in + i] = w[(o * in + i) * kernel + k];
    return Append(packed);
  }

  ResidualUnit Unit(const std::string& prefix, std::size_t channels,
                    std::size_t dilation) {
    ResidualUnit unit;
    unit.dilation = dilation;
    unit.alpha_in = Raw(prefix + ".block.0.alpha", {1, channels, 1});
    unit.depthwise =
        Append(Normalized(prefix + ".block.1", {channels, 1, kKernel}));
    unit.depthwise_bias = Raw(prefix + ".block.1.bias", {channels});
    unit.alpha_out = Raw(prefix + ".block.2.alpha", {1, channels, 1});
    unit.pointwise =
        Append(Normalized(prefix + ".block.3", {channels, channels, 1}));
    unit.pointwise_bias = Raw(prefix + ".block.3.bias", {channels});
    return unit;
  }

private:
  const ModelFiles& files_;
  std::vector<float> data_;
};

}  // namespace

std::unique_ptr<AudioVae> AudioVae::Create(const Config& config,
                                           const ModelFiles& files,
                                           std::string* error) {
  std::unique_ptr<AudioVae> vae(new AudioVae());
  try {
    vae->config_ = config.vae;
    const AudioVaeConfig& c = config.vae;
    Packer pack(files);
    auto weights = std::make_unique<Weights>();
    Weights& w = *weights;

    const std::size_t latent = c.latent_dim;
    w.in_depthwise =
        pack.Append(pack.Normalized("decoder.model.0", {latent, 1, kKernel}));
    w.in_depthwise_bias = pack.Raw("decoder.model.0.bias", {latent});
    w.in_pointwise = pack.Append(
        pack.Normalized("decoder.model.1", {c.decoder_dim, latent, 1}));
    w.in_pointwise_bias = pack.Raw("decoder.model.1.bias", {c.decoder_dim});
    const std::size_t buckets = c.sr_bin_boundaries.size() + 1;
    const std::size_t bucket = c.rate_bucket();
    std::size_t channels = c.decoder_dim;
    for (std::size_t index = 0; index < c.decoder_rates.size(); ++index) {
      DecoderBlock block;
      block.in = channels;
      block.out = channels / 2;
      block.stride = c.decoder_rates[index];
      const std::string layer = "decoder.model." + std::to_string(index + 2);
      const std::string cond =
          "decoder.sr_cond_model." + std::to_string(index + 2);
      const std::vector<float> scale =
          pack.Get(cond + ".scale_embed.weight", {buckets, block.in});
      const std::vector<float> bias =
          pack.Get(cond + ".bias_embed.weight", {buckets, block.in});
      block.scale = pack.Append(
          std::vector<float>(scale.begin() + bucket * block.in,
                             scale.begin() + (bucket + 1) * block.in));
      block.bias = pack.Append(
          std::vector<float>(bias.begin() + bucket * block.in,
                             bias.begin() + (bucket + 1) * block.in));
      block.alpha = pack.Raw(layer + ".block.0.alpha", {1, block.in, 1});
      const std::size_t s = block.stride;
      // ConvTranspose1d weight [in][out][2s]; weight_norm's dim 0 is `in`.
      const std::vector<float> up =
          pack.Normalized(layer + ".block.1", {block.in, block.out, 2 * s});
      std::vector<float> current(s * block.out * block.in);
      std::vector<float> previous(current.size());
      for (std::size_t i = 0; i < block.in; ++i)
        for (std::size_t o = 0; o < block.out; ++o)
          for (std::size_t r = 0; r < s; ++r) {
            const float* taps = up.data() + (i * block.out + o) * 2 * s;
            current[(r * block.out + o) * block.in + i] = taps[r];
            previous[(r * block.out + o) * block.in + i] = taps[r + s];
          }
      block.up_current = pack.Append(current);
      block.up_previous = pack.Append(previous);
      const std::vector<float> up_bias =
          pack.Get(layer + ".block.1.bias", {block.out});
      std::vector<float> repeated;
      for (std::size_t r = 0; r < s; ++r)
        repeated.insert(repeated.end(), up_bias.begin(), up_bias.end());
      block.up_bias = pack.Append(repeated);
      for (std::size_t u = 0; u < 3; ++u)
        block.units[u] = pack.Unit(layer + ".block." + std::to_string(u + 2),
                                   block.out, kDilations[u]);
      w.blocks.push_back(block);
      channels = block.out;
    }
    const std::size_t last = c.decoder_rates.size() + 2;
    w.out_channels = channels;
    w.out_alpha = pack.Raw("decoder.model." + std::to_string(last) + ".alpha",
                           {1, channels, 1});
    const std::string out_conv = "decoder.model." + std::to_string(last + 1);
    w.out_conv = pack.Append(pack.Normalized(out_conv, {1, channels, kKernel}));
    w.out_bias = pack.Get(out_conv + ".bias", {1})[0];

    w.encoder_in = pack.Append(
        pack.Normalized("encoder.block.0", {c.encoder_dim, 1, kKernel}));
    w.encoder_in_bias = pack.Raw("encoder.block.0.bias", {c.encoder_dim});
    channels = c.encoder_dim;
    for (std::size_t index = 0; index < c.encoder_rates.size(); ++index) {
      EncoderBlock block;
      block.in = channels;
      block.out = 2 * channels;
      block.stride = c.encoder_rates[index];
      const std::string layer = "encoder.block." + std::to_string(index + 1);
      for (std::size_t u = 0; u < 3; ++u)
        block.units[u] = pack.Unit(layer + ".block." + std::to_string(u),
                                   block.in, kDilations[u]);
      block.alpha = pack.Raw(layer + ".block.3.alpha", {1, block.in, 1});
      block.down = pack.PerTap(layer + ".block.4", block.out, block.in,
                               2 * block.stride);
      block.down_bias = pack.Raw(layer + ".block.4.bias", {block.out});
      w.encoder_blocks.push_back(block);
      channels = block.out;
    }
    w.mu_channels = channels;
    w.mu = pack.PerTap("encoder.fc_mu", latent, channels, 3);
    w.mu_bias = pack.Raw("encoder.fc_mu.bias", {latent});

    const std::vector<float>& data = pack.data();
    vae->storage_ = DeviceBuffer<float>(data.size());
    Check(hipMemcpy(vae->storage_.get(), data.data(),
                    data.size() * sizeof(float), hipMemcpyHostToDevice),
          "hipMemcpy AudioVAE");
    vae->weights_ = std::move(weights);

    Check(hipStreamCreateWithFlags(&vae->stream_, hipStreamNonBlocking),
          "hipStreamCreate");
    Check(hipblasCreate(&vae->blas_), "hipblasCreate");
    Check(hipblasSetStream(vae->blas_, vae->stream_), "hipblasSetStream");

    // Streaming state, in decoding order: the input convolution, then per
    // block the transposed convolution's last frame and the residual units'
    // history, then the output convolution.
    vae->states_.emplace_back((kKernel - 1) * latent);
    for (const DecoderBlock& block : w.blocks) {
      vae->states_.emplace_back(block.in);
      for (const ResidualUnit& unit : block.units)
        vae->states_.emplace_back((kKernel - 1) * unit.dilation * block.out);
    }
    vae->states_.emplace_back((kKernel - 1) * w.out_channels);
    vae->Reset();
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return nullptr;
  }
  return vae;
}

AudioVae::~AudioVae() {
  if (stream_ != nullptr)
    (void)hipStreamSynchronize(stream_);
  if (pinned_ != nullptr)
    (void)hipHostFree(pinned_);
  if (blas_ != nullptr)
    (void)hipblasDestroy(blas_);
  if (stream_ != nullptr)
    (void)hipStreamDestroy(stream_);
}

void AudioVae::Reset() {
  for (DeviceBuffer<float>& state : states_)
    Check(hipMemsetAsync(state.get(), 0, state.size() * sizeof(float), stream_),
          "hipMemset state");
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
}

void AudioVae::Gemm(const float* x, std::size_t x_stride, const float* w,
                    float* y, std::size_t rows, std::size_t in, std::size_t out,
                    bool accumulate) {
  const float alpha = 1.0F;
  const float beta = accumulate ? 1.0F : 0.0F;
  Check(hipblasSgemm(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(out),
                     static_cast<int>(rows), static_cast<int>(in), &alpha, w,
                     static_cast<int>(in), x, static_cast<int>(x_stride), &beta,
                     y, static_cast<int>(out)),
        "hipblasSgemm");
}

void AudioVae::Decode(const float* latents, std::size_t frames,
                      std::vector<float>* audio) {
  if (frames == 0)
    return;
  latents_.Reserve(frames * config_.latent_dim);
  Check(hipMemcpyAsync(latents_.get(), latents,
                       frames * config_.latent_dim * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy latents");
  Run(frames);
  Collect(audio);
}

void AudioVae::Enqueue(const float* latents, std::size_t frames,
                       hipEvent_t ready) {
  if (frames == 0)
    return;
  if (ready != nullptr)
    Check(hipStreamWaitEvent(stream_, ready, 0), "hipStreamWaitEvent");
  latents_.Reserve(frames * config_.latent_dim);
  Check(hipMemcpyAsync(latents_.get(), latents,
                       frames * config_.latent_dim * sizeof(float),
                       hipMemcpyDeviceToDevice, stream_),
        "hipMemcpy latents");
  Run(frames);
}

void AudioVae::Collect(std::vector<float>* audio) {
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  audio->insert(audio->end(), pinned_, pinned_ + pending_);
  pending_ = 0;
}

void AudioVae::Run(std::size_t frames) {
  const Weights& w = *weights_;
  const float* base = storage_.get();
  const auto at = [base](Offset offset) { return base + offset; };
  const std::size_t latent = config_.latent_dim;
  // The widest stage: frames * (product of strides so far) * channels.
  std::size_t widest = frames * std::max(latent, config_.decoder_dim);
  std::size_t length = frames;
  for (const DecoderBlock& block : w.blocks) {
    length *= block.stride;
    widest = std::max(widest, length * block.out);
  }
  const std::size_t history = (kKernel - 1) * kDilations.back();
  hidden_.Reserve(widest);
  scratch_.Reserve(widest);
  work_.Reserve(widest + history * config_.decoder_dim);
  samples_.Reserve(length);
  if (pinned_capacity_ < length) {
    if (pinned_ != nullptr)
      Check(hipHostFree(pinned_), "hipHostFree");
    pinned_ = nullptr;
    Check(hipHostMalloc(reinterpret_cast<void**>(&pinned_),
                        length * sizeof(float)),
          "hipHostMalloc");
    pinned_capacity_ = length;
  }

  std::size_t state = 0;
  // Copies the state in front of `work`, and afterwards keeps the last
  // `rows` rows of state ++ input as the new state.
  const auto load = [&](std::size_t rows, std::size_t width) {
    Check(hipMemcpyAsync(work_.get(), states_[state].get(),
                         rows * width * sizeof(float), hipMemcpyDeviceToDevice,
                         stream_),
          "hipMemcpy state");
  };
  const auto keep = [&](std::size_t rows, std::size_t width,
                        std::size_t frames_in) {
    Check(hipMemcpyAsync(states_[state].get(), work_.get() + frames_in * width,
                         rows * width * sizeof(float), hipMemcpyDeviceToDevice,
                         stream_),
          "hipMemcpy state");
    ++state;
  };

  load(kKernel - 1, latent);
  Check(hipMemcpyAsync(work_.get() + (kKernel - 1) * latent, latents_.get(),
                       frames * latent * sizeof(float), hipMemcpyDeviceToDevice,
                       stream_),
        "hipMemcpy latents");
  kernels::DepthwiseConv(work_.get(), at(w.in_depthwise),
                         at(w.in_depthwise_bias), scratch_.get(), frames,
                         latent, kKernel, 1, stream_);
  keep(kKernel - 1, latent, frames);
  Gemm(scratch_.get(), latent, at(w.in_pointwise), hidden_.get(), frames,
       latent, config_.decoder_dim, false);
  kernels::AddBiasF32(hidden_.get(), at(w.in_pointwise_bias), frames,
                      config_.decoder_dim, stream_);

  std::size_t t = frames;
  for (const DecoderBlock& block : w.blocks) {
    // Snake of the rate-conditioned input, after one frame of history.
    load(1, block.in);
    kernels::Snake(hidden_.get(), work_.get() + block.in, t, block.in,
                   at(block.alpha), at(block.scale), at(block.bias), stream_);
    keep(1, block.in, t);
    const std::size_t width = block.stride * block.out;
    Gemm(work_.get() + block.in, block.in, at(block.up_current), hidden_.get(),
         t, block.in, width, false);
    Gemm(work_.get(), block.in, at(block.up_previous), hidden_.get(), t,
         block.in, width, true);
    kernels::AddBiasF32(hidden_.get(), at(block.up_bias), t, width, stream_);
    t *= block.stride;
    const std::size_t c = block.out;
    for (const ResidualUnit& unit : block.units) {
      const std::size_t rows = (kKernel - 1) * unit.dilation;
      load(rows, c);
      kernels::Snake(hidden_.get(), work_.get() + rows * c, t, c,
                     at(unit.alpha_in), nullptr, nullptr, stream_);
      keep(rows, c, t);
      kernels::DepthwiseConv(work_.get(), at(unit.depthwise),
                             at(unit.depthwise_bias), scratch_.get(), t, c,
                             kKernel, unit.dilation, stream_);
      kernels::Snake(scratch_.get(), scratch_.get(), t, c, at(unit.alpha_out),
                     nullptr, nullptr, stream_);
      Gemm(scratch_.get(), c, at(unit.pointwise), hidden_.get(), t, c, c, true);
      kernels::AddBiasF32(hidden_.get(), at(unit.pointwise_bias), t, c,
                          stream_);
    }
  }
  const std::size_t c = w.out_channels;
  load(kKernel - 1, c);
  kernels::Snake(hidden_.get(), work_.get() + (kKernel - 1) * c, t, c,
                 at(w.out_alpha), nullptr, nullptr, stream_);
  keep(kKernel - 1, c, t);
  kernels::ConvToMono(work_.get(), at(w.out_conv), w.out_bias, samples_.get(),
                      t, c, kKernel, stream_);
  Check(hipMemcpyAsync(pinned_, samples_.get(), t * sizeof(float),
                       hipMemcpyDeviceToHost, stream_),
        "hipMemcpy audio");
  pending_ = t;
}

std::vector<float> AudioVae::Encode(const std::vector<float>& samples) {
  const Weights& w = *weights_;
  const float* base = storage_.get();
  const auto at = [base](Offset offset) { return base + offset; };
  const std::size_t hop = config_.hop();
  if (samples.empty() || samples.size() % hop != 0)
    throw std::invalid_argument("AudioVAE input is not a multiple of its hop");
  std::size_t t = samples.size();
  std::size_t widest = t * config_.encoder_dim;
  {
    std::size_t length = t;
    for (const EncoderBlock& block : w.encoder_blocks) {
      widest = std::max(widest, length * block.in);
      length /= block.stride;
      widest = std::max(widest, length * block.out);
    }
  }
  const std::size_t history = (kKernel - 1) * kDilations.back();
  hidden_.Reserve(widest);
  scratch_.Reserve(widest);
  work_.Reserve(widest + history * w.mu_channels);
  samples_.Reserve(t + kKernel - 1);

  Check(
      hipMemsetAsync(samples_.get(), 0, (kKernel - 1) * sizeof(float), stream_),
      "hipMemset");
  Check(hipMemcpyAsync(samples_.get() + kKernel - 1, samples.data(),
                       t * sizeof(float), hipMemcpyHostToDevice, stream_),
        "hipMemcpy samples");
  kernels::ConvFromMono(samples_.get(), at(w.encoder_in), at(w.encoder_in_bias),
                        hidden_.get(), t, config_.encoder_dim, kKernel,
                        stream_);
  for (const EncoderBlock& block : w.encoder_blocks) {
    const std::size_t c = block.in;
    for (const ResidualUnit& unit : block.units) {
      const std::size_t rows = (kKernel - 1) * unit.dilation;
      Check(hipMemsetAsync(work_.get(), 0, rows * c * sizeof(float), stream_),
            "hipMemset");
      kernels::Snake(hidden_.get(), work_.get() + rows * c, t, c,
                     at(unit.alpha_in), nullptr, nullptr, stream_);
      kernels::DepthwiseConv(work_.get(), at(unit.depthwise),
                             at(unit.depthwise_bias), scratch_.get(), t, c,
                             kKernel, unit.dilation, stream_);
      kernels::Snake(scratch_.get(), scratch_.get(), t, c, at(unit.alpha_out),
                     nullptr, nullptr, stream_);
      Gemm(scratch_.get(), c, at(unit.pointwise), hidden_.get(), t, c, c, true);
      kernels::AddBiasF32(hidden_.get(), at(unit.pointwise_bias), t, c,
                          stream_);
    }
    // Strided causal convolution (kernel 2s, left padding s): one product
    // per tap over every s-th frame.
    const std::size_t s = block.stride;
    Check(hipMemsetAsync(work_.get(), 0, s * c * sizeof(float), stream_),
          "hipMemset");
    kernels::Snake(hidden_.get(), work_.get() + s * c, t, c, at(block.alpha),
                   nullptr, nullptr, stream_);
    const std::size_t frames = t / s;
    for (std::size_t k = 0; k < 2 * s; ++k)
      Gemm(work_.get() + k * c, s * c, at(block.down) + k * block.out * c,
           hidden_.get(), frames, c, block.out, k > 0);
    kernels::AddBiasF32(hidden_.get(), at(block.down_bias), frames, block.out,
                        stream_);
    t = frames;
  }
  // fc_mu: kernel 3, left padding 2.
  const std::size_t c = w.mu_channels;
  const std::size_t latent = config_.latent_dim;
  Check(hipMemsetAsync(work_.get(), 0, 2 * c * sizeof(float), stream_),
        "hipMemset");
  Check(hipMemcpyAsync(work_.get() + 2 * c, hidden_.get(),
                       t * c * sizeof(float), hipMemcpyDeviceToDevice, stream_),
        "hipMemcpy");
  for (std::size_t k = 0; k < 3; ++k)
    Gemm(work_.get() + k * c, c, at(w.mu) + k * latent * c, scratch_.get(), t,
         c, latent, k > 0);
  kernels::AddBiasF32(scratch_.get(), at(w.mu_bias), t, latent, stream_);
  std::vector<float> result(t * latent);
  Check(hipMemcpyAsync(result.data(), scratch_.get(),
                       result.size() * sizeof(float), hipMemcpyDeviceToHost,
                       stream_),
        "hipMemcpy latents");
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  return result;
}

}  // namespace gufo::models::voxcpm
