#ifndef GUFO_MODELS_VOXCPM_VAE_HPP_
#define GUFO_MODELS_VOXCPM_VAE_HPP_

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "device.hpp"
#include "model_files.hpp"

namespace gufo::models::voxcpm {

/// AudioVAE V2 (`audiovae.pth`) on the GPU in float32: the causal decoder
/// from latents to `out_sample_rate` audio, run chunk by chunk with the
/// state of every causal convolution carried between chunks (the
/// reference's StreamingVAEDecoder, which equals decoding at once), and
/// the causal encoder from `sample_rate` audio to latents for cloning.
/// Weight normalization is folded into the weights at load time.
class AudioVae {
public:
  [[nodiscard]] static std::unique_ptr<AudioVae> Create(
      const Config& config, const ModelFiles& files,
      std::string* error = nullptr);
  ~AudioVae();
  AudioVae(const AudioVae&) = delete;
  AudioVae& operator=(const AudioVae&) = delete;

  /// Starts a new decoded stream (zero history).
  void Reset();
  /// Decodes `frames` latent frames ([frames][latent_dim], time-major,
  /// on the host) and appends frames * decode_hop samples to `audio`.
  void Decode(const float* latents, std::size_t frames,
              std::vector<float>* audio);
  /// Queues the decoding of device latents once `ready` (may be null) has
  /// completed, on the AudioVAE's own stream; the latents are copied before
  /// anything else. Collect() appends the samples; one decoding may be
  /// pending at a time.
  void Enqueue(const float* latents, std::size_t frames, hipEvent_t ready);
  void Collect(std::vector<float>* audio);
  /// Encodes mono audio at `sample_rate` whose length is a multiple of the
  /// hop; returns [frames][latent_dim] (the posterior mean).
  [[nodiscard]] std::vector<float> Encode(const std::vector<float>& samples);

  [[nodiscard]] std::size_t output_rate() const {
    return config_.out_sample_rate;
  }

  struct Weights;

private:
  AudioVae() = default;
  /// Decodes latents_ ([frames][latent_dim]) into the pinned buffer.
  void Run(std::size_t frames);

  /// y[rows][out] (+)= x[rows][in] (row stride `x_stride`) W^T.
  void Gemm(const float* x, std::size_t x_stride, const float* w, float* y,
            std::size_t rows, std::size_t in, std::size_t out, bool accumulate);

  AudioVaeConfig config_;
  hipStream_t stream_ = nullptr;
  hipblasHandle_t blas_ = nullptr;
  std::unique_ptr<Weights> weights_;
  DeviceBuffer<float> storage_;
  // Streaming state of each causal convolution, in decoding order.
  std::vector<DeviceBuffer<float>> states_;
  DeviceBuffer<float> latents_;
  DeviceBuffer<float> work_;
  DeviceBuffer<float> hidden_;
  DeviceBuffer<float> scratch_;
  DeviceBuffer<float> samples_;
  float* pinned_ = nullptr;
  std::size_t pinned_capacity_ = 0;
  std::size_t pending_ = 0;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_VAE_HPP_
