#ifndef GUFO_MODELS_VOXCPM_GENERATOR_HPP_
#define GUFO_MODELS_VOXCPM_GENERATOR_HPP_

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "device.hpp"
#include "gemm.hpp"
#include "model_files.hpp"
#include "transformer.hpp"
#include "weights.hpp"

namespace gufo::models::voxcpm {

/// The prompt rows of VoxCPM2Model._inference: per row a text token or an
/// audio latent patch.
struct Prompt {
  std::vector<std::int32_t> tokens;
  std::vector<std::int32_t> text_mask;
  std::vector<std::int32_t> audio_mask;
  /// [rows][patch_size][feat_dim]; zero on text rows.
  std::vector<float> feats;
};

/// The Euler steps of UnifiedCFM.solve_euler with the sway-sampled
/// schedule, in the reference's float32 arithmetic: the time and step of
/// each update, and whether it evaluates the DiT (CFG-Zero* leaves the
/// first max(1, int(0.04 * (n + 1))) steps at zero velocity).
struct EulerStep {
  float t = 0.0F;
  float dt = 0.0F;
  bool evaluate = true;
};
[[nodiscard]] std::vector<EulerStep> EulerSchedule(std::size_t timesteps,
                                                   float sway = 1.0F);

/// Intermediate tensors for tests, copied to the host when tracing.
struct Trace {
  std::vector<float> lm_prefill;                  // [rows][hidden], before FSQ
  std::vector<float> residual_prefill;            // [rows][hidden]
  std::vector<std::vector<float>> mu;             // per step [2 * dit hidden]
  std::vector<std::vector<float>> lm_step;        // before FSQ
  std::vector<std::vector<float>> residual_step;  // per step
};

/// VoxCPM2's autoregressive generator on the GPU: the text-semantic LM, the
/// residual acoustic LM with their KV caches, the local encoder, the local
/// DiT with its CFM sampler and the stop predictor. One sequence at a time.
class Generator {
public:
  [[nodiscard]] static std::unique_ptr<Generator> Create(
      const Config& config, const ModelFiles& files,
      std::string* error = nullptr);
  ~Generator();
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  [[nodiscard]] const Config& config() const { return config_; }
  /// Prompt rows plus generated patches the caches can hold.
  [[nodiscard]] std::size_t capacity() const { return config_.max_length; }
  /// Rows in the caches (the next patch's position).
  [[nodiscard]] std::size_t position() const { return position_; }

  /// Starts a sequence from `prompt` (at most capacity() - 1 rows).
  void Prefill(const Prompt& prompt);

  /// Generates the next patch from `noise` ([patch_size][feat_dim]) with
  /// `timesteps` Euler steps and guidance `cfg`: writes the stop logits of
  /// the state before it (and its latent, [patch_size][feat_dim], when
  /// `latent` isn't null), then advances both LMs past it. The latent also
  /// stays on the device: latent() until the step after next, ready once
  /// latent_event() completes.
  void Step(const float* noise, std::size_t timesteps, float cfg, float* latent,
            float* stop_logits);

  [[nodiscard]] const float* latent() const { return latent_slot_; }
  [[nodiscard]] hipEvent_t latent_event() const { return latent_event_; }

  /// The DiT's CFG pair of velocities [2][patch_size][feat_dim] for one
  /// evaluation at time `t` (tests).
  void Velocity(const float* mu, const float* cond, const float* x, float t,
                float* velocity);

  /// Copies intermediate tensors into `trace` while set (tests).
  void set_trace(Trace* trace) { trace_ = trace; }

private:
  Generator() = default;

  const float* TimeVector(float t);
  void Encode(const float* latents, std::size_t patches, float* embeds);
  void EvaluateDit(const float* x, float t, float* velocity);
  void Fsq(const float* input, float* output, std::size_t rows);
  void Upload(DeviceBuffer<std::int32_t>& buffer,
              const std::vector<std::int32_t>& values);
  std::vector<float> Download(const float* data, std::size_t count);

  Config config_;
  hipStream_t stream_ = nullptr;
  hipblasHandle_t blas_ = nullptr;
  std::unique_ptr<Bf16Gemm> gemm_;
  std::unique_ptr<DeviceWeights> device_weights_;
  const GeneratorWeights* w_ = nullptr;
  DeviceBuffer<float> rope_;
  KvCache base_cache_;
  KvCache residual_cache_;
  KvCache encoder_cache_;
  KvCache dit_cache_;
  StackScratch scratch_;
  std::size_t position_ = 0;

  // Sequence state.
  DeviceBuffer<float> lm_hidden_;        // [hidden], after FSQ
  DeviceBuffer<float> residual_hidden_;  // [hidden]
  DeviceBuffer<float> prefix_cond_;      // [patch][feat]
  DeviceBuffer<float> x_;                // [patch][feat]
  DeviceBuffer<float> velocity_;         // [2][patch][feat]
  DeviceBuffer<float> mu_;               // [2 * dit hidden]
  DeviceBuffer<float> cond_rows_;        // [patch][dit hidden]
  DeviceBuffer<float> stop_;             // [2]
  DeviceBuffer<float> delta_time_;       // delta_time_mlp(sinusoid(0))
  std::map<float, DeviceBuffer<float>> time_vectors_;

  // Index arrays.
  DeviceBuffer<std::int32_t> dit_positions_;
  DeviceBuffer<std::int32_t> dit_rows_;
  DeviceBuffer<std::int32_t> dit_outputs_;
  DeviceBuffer<std::int32_t> ids_;
  DeviceBuffer<std::int32_t> text_mask_;
  DeviceBuffer<std::int32_t> audio_mask_;
  DeviceBuffer<std::int32_t> audio_rows_;
  DeviceBuffer<std::int32_t> positions_;
  DeviceBuffer<std::int32_t> cache_rows_;
  DeviceBuffer<std::int32_t> local_positions_;
  DeviceBuffer<std::int32_t> local_rows_;
  DeviceBuffer<std::int32_t> local_firsts_;
  std::size_t encoder_rows_count_ = 0;

  // Working buffers.
  DeviceBuffer<Bf16> encoder_bf16_;
  DeviceBuffer<float> encoder_a_;
  DeviceBuffer<float> encoder_b_;
  DeviceBuffer<float> encoder_c_;
  DeviceBuffer<Bf16> fsq_bf16_;
  DeviceBuffer<float> fsq_latent_;
  DeviceBuffer<Bf16> dit_bf16_;
  DeviceBuffer<float> x_rows_;
  DeviceBuffer<float> dit_hidden_;
  DeviceBuffer<float> dit_output_;
  DeviceBuffer<float> prefill_a_;
  DeviceBuffer<float> prefill_b_;
  DeviceBuffer<float> prefill_c_;
  DeviceBuffer<float> prefill_d_;
  DeviceBuffer<Bf16> prefill_bf16_;
  DeviceBuffer<float> prompt_latents_;
  DeviceBuffer<float> prompt_embeds_;
  DeviceBuffer<Bf16> step_bf16_;
  DeviceBuffer<float> step_a_;
  DeviceBuffer<float> step_b_;
  DeviceBuffer<float> embed_;
  // The last two patches' latents, and the pinned staging of each step's
  // noise and cache position (written only after the previous step's
  // final synchronization).
  DeviceBuffer<float> latent_ring_;
  const float* latent_slot_ = nullptr;
  hipEvent_t latent_event_ = nullptr;
  float* pinned_noise_ = nullptr;
  std::int32_t* pinned_position_ = nullptr;
  Trace* trace_ = nullptr;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_GENERATOR_HPP_
