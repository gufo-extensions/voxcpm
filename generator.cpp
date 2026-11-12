// Ported from OpenBMB/VoxCPM (Apache-2.0, Copyright OpenBMB; see
// licenses/VoxCPM.txt): VoxCPM2Model's generation step
// (src/voxcpm/model/voxcpm2.py), the local encoder and local DiT
// (src/voxcpm/modules/locenc, locdit), the scalar quantization layer and
// UnifiedCFM's Euler sampler. VoxCPM credits CosyVoice (Apache-2.0, Copyright
// Alibaba Inc; see licenses/CosyVoice.txt) for its flow-matching local DiT.
// Rewritten in C++ and HIP for Gufo.

#include "generator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "kernels/generator_kernels.hpp"

namespace gufo::models::voxcpm {
namespace {

/// torch.linspace in float32: the first half counts up from `start`, the
/// second half down from `end`.
std::vector<float> Linspace(float start, float end, std::size_t steps) {
  std::vector<float> values(steps);
  if (steps == 1) {
    values[0] = start;
    return values;
  }
  const float step = (end - start) / static_cast<float>(steps - 1);
  const std::size_t halfway = steps / 2;
  for (std::size_t index = 0; index < steps; ++index) {
    values[index] = index < halfway
                        ? start + step * static_cast<float>(index)
                        : end - step * static_cast<float>(steps - index - 1);
  }
  return values;
}

/// SinusoidalPosEmb(width)(t) with scale 1000, in float32.
std::vector<float> Sinusoid(float t, std::size_t width) {
  const std::size_t half = width / 2;
  const auto exponent =
      static_cast<float>(std::log(10000.0) / static_cast<double>(half - 1));
  std::vector<float> embedding(width);
  const float scaled = 1000.0F * t;
  for (std::size_t i = 0; i < half; ++i) {
    const float frequency = std::exp(static_cast<float>(i) * -exponent);
    const float angle = scaled * frequency;
    embedding[i] = std::sin(angle);
    embedding[half + i] = std::cos(angle);
  }
  return embedding;
}

std::vector<std::int32_t> Iota(std::size_t count) {
  std::vector<std::int32_t> values(count);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

}  // namespace

std::vector<EulerStep> EulerSchedule(std::size_t timesteps, float sway) {
  std::vector<float> span = Linspace(1.0F, 0.0F, timesteps + 1);
  const auto quarter_turn = static_cast<float>(M_PI / 2.0);
  for (float& t : span)
    t = t + sway * (std::cos(quarter_turn * t) - 1.0F + t);
  const auto zero_steps = std::max<std::size_t>(
      1, static_cast<std::size_t>(static_cast<double>(span.size()) * 0.04));
  std::vector<EulerStep> steps;
  float t = span[0];
  float dt = span[0] - span[1];
  for (std::size_t step = 1; step < span.size(); ++step) {
    steps.push_back({t, dt, step > zero_steps});
    t = t - dt;
    if (step < span.size() - 1)
      dt = t - span[step + 1];
  }
  return steps;
}

std::unique_ptr<Generator> Generator::Create(const Config& config,
                                             const ModelFiles& files,
                                             std::string* error) {
  std::unique_ptr<Generator> generator(new Generator());
  try {
    generator->config_ = config;
    Check(hipStreamCreateWithFlags(&generator->stream_, hipStreamNonBlocking),
          "hipStreamCreate");
    Check(hipblasCreate(&generator->blas_), "hipblasCreate");
    Check(hipblasSetStream(generator->blas_, generator->stream_),
          "hipblasSetStream");
    generator->gemm_ =
        std::make_unique<Bf16Gemm>(generator->blas_, generator->stream_);
    generator->device_weights_ = DeviceWeights::Upload(files, config, error);
    if (generator->device_weights_ == nullptr)
      return nullptr;
    generator->w_ = &generator->device_weights_->weights();

    const std::vector<float> rope =
        RopeTable(config.rope, std::max<std::size_t>(config.max_length, 64));
    generator->rope_ = DeviceBuffer<float>(rope.size());
    Check(hipMemcpy(generator->rope_.get(), rope.data(),
                    rope.size() * sizeof(float), hipMemcpyHostToDevice),
          "hipMemcpy rope");
    generator->base_cache_ = KvCache(config.lm, config.max_length);
    generator->residual_cache_ = KvCache(config.residual_lm, config.max_length);
    const std::size_t dit_rows = 2 * (3 + 2 * config.patch_size);
    generator->dit_cache_ = KvCache(config.dit, dit_rows);

    // The DiT's rows: two sequences of 3 + 2P positions; the outputs are
    // the last P rows of each.
    const std::size_t length = dit_rows / 2;
    std::vector<std::int32_t> positions(dit_rows);
    std::vector<std::int32_t> outputs;
    for (std::size_t row = 0; row < dit_rows; ++row)
      positions[row] = static_cast<std::int32_t>(row % length);
    for (std::size_t sequence = 0; sequence < 2; ++sequence)
      for (std::size_t i = 0; i < config.patch_size; ++i)
        outputs.push_back(static_cast<std::int32_t>(sequence * length + 3 +
                                                    config.patch_size + i));
    generator->Upload(generator->dit_positions_, positions);
    generator->Upload(generator->dit_rows_, Iota(dit_rows));
    generator->Upload(generator->dit_outputs_, outputs);

    const std::size_t hidden = config.lm.hidden;
    generator->lm_hidden_ = DeviceBuffer<float>(hidden);
    generator->residual_hidden_ = DeviceBuffer<float>(hidden);
    generator->prefix_cond_ =
        DeviceBuffer<float>(config.patch_size * config.feat_dim);
    generator->x_ = DeviceBuffer<float>(config.patch_size * config.feat_dim);
    generator->velocity_ =
        DeviceBuffer<float>(2 * config.patch_size * config.feat_dim);
    generator->mu_ = DeviceBuffer<float>(2 * config.dit.hidden);
    generator->cond_rows_ =
        DeviceBuffer<float>(config.patch_size * config.dit.hidden);
    generator->stop_ = DeviceBuffer<float>(2);
    generator->latent_ring_ =
        DeviceBuffer<float>(2 * config.patch_size * config.feat_dim);
    Check(hipEventCreateWithFlags(&generator->latent_event_,
                                  hipEventDisableTiming),
          "hipEventCreate");
    Check(hipHostMalloc(reinterpret_cast<void**>(&generator->pinned_noise_),
                        config.patch_size * config.feat_dim * sizeof(float)),
          "hipHostMalloc");
    Check(hipHostMalloc(reinterpret_cast<void**>(&generator->pinned_position_),
                        sizeof(std::int32_t)),
          "hipHostMalloc");

    // delta_time_mlp(sinusoid(0)): dt is zero outside mean mode.
    const std::vector<float> zero = Sinusoid(0.0F, config.dit.hidden);
    DeviceBuffer<float> input(zero.size());
    Check(hipMemcpy(input.get(), zero.data(), zero.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "hipMemcpy time");
    DeviceBuffer<Bf16> input_bf16(zero.size());
    DeviceBuffer<float> middle(zero.size());
    generator->delta_time_ = DeviceBuffer<float>(zero.size());
    const GeneratorWeights& w = *generator->w_;
    hipStream_t stream = generator->stream_;
    kernels::ToBf16(input.get(), input_bf16.get(), zero.size(), false, stream);
    Project(w.delta_time_1, input_bf16.get(), middle.get(), 1,
            *generator->gemm_, stream);
    kernels::ToBf16(middle.get(), input_bf16.get(), zero.size(), true, stream);
    Project(w.delta_time_2, input_bf16.get(), generator->delta_time_.get(), 1,
            *generator->gemm_, stream);
    Check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return nullptr;
  }
  return generator;
}

Generator::~Generator() {
  if (stream_ != nullptr)
    (void)hipStreamSynchronize(stream_);
  if (latent_event_ != nullptr)
    (void)hipEventDestroy(latent_event_);
  if (pinned_noise_ != nullptr)
    (void)hipHostFree(pinned_noise_);
  if (pinned_position_ != nullptr)
    (void)hipHostFree(pinned_position_);
  gemm_.reset();
  if (blas_ != nullptr)
    (void)hipblasDestroy(blas_);
  if (stream_ != nullptr)
    (void)hipStreamDestroy(stream_);
}

void Generator::Upload(DeviceBuffer<std::int32_t>& buffer,
                       const std::vector<std::int32_t>& values) {
  buffer.Reserve(values.size());
  Check(hipMemcpyAsync(buffer.get(), values.data(),
                       values.size() * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy indices");
  // The host vector may go away; wait for the copy.
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
}

std::vector<float> Generator::Download(const float* data, std::size_t count) {
  std::vector<float> values(count);
  Check(hipMemcpyAsync(values.data(), data, count * sizeof(float),
                       hipMemcpyDeviceToHost, stream_),
        "hipMemcpy download");
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  return values;
}

const float* Generator::TimeVector(float t) {
  const auto found = time_vectors_.find(t);
  if (found != time_vectors_.end())
    return found->second.get();
  const std::size_t width = config_.dit.hidden;
  const std::vector<float> sinusoid = Sinusoid(t, width);
  DeviceBuffer<float> input(width);
  DeviceBuffer<Bf16> input_bf16(width);
  DeviceBuffer<float> middle(width);
  DeviceBuffer<float> vector(width);
  Check(hipMemcpyAsync(input.get(), sinusoid.data(), width * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy time");
  // time_mlp(sinusoid(t)) + delta_time_mlp(sinusoid(0)).
  kernels::ToBf16(input.get(), input_bf16.get(), width, false, stream_);
  Project(w_->time_1, input_bf16.get(), middle.get(), 1, *gemm_, stream_);
  kernels::ToBf16(middle.get(), input_bf16.get(), width, true, stream_);
  Check(hipMemcpyAsync(vector.get(), delta_time_.get(), width * sizeof(float),
                       hipMemcpyDeviceToDevice, stream_),
        "hipMemcpy time");
  gemm_->Linear(input_bf16.get(), w_->time_2.weight, vector.get(), 1, width,
                width, true);
  kernels::AddBias(vector.get(), w_->time_2.bias, 1, width, stream_);
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  return time_vectors_.emplace(t, std::move(vector)).first->second.get();
}

void Generator::Fsq(const float* input, float* output, std::size_t rows) {
  const std::size_t hidden = config_.lm.hidden;
  const std::size_t latent = config_.fsq_latent_dim;
  fsq_bf16_.Reserve(rows * std::max(hidden, latent));
  fsq_latent_.Reserve(rows * latent);
  kernels::ToBf16(input, fsq_bf16_.get(), rows * hidden, false, stream_);
  Project(w_->fsq_in, fsq_bf16_.get(), fsq_latent_.get(), rows, *gemm_,
          stream_);
  kernels::Quantize(fsq_latent_.get(), fsq_bf16_.get(), rows * latent,
                    config_.fsq_scale, stream_);
  Project(w_->fsq_out, fsq_bf16_.get(), output, rows, *gemm_, stream_);
}

void Generator::Encode(const float* latents, std::size_t patches,
                       float* embeds) {
  const std::size_t patch = config_.patch_size;
  const std::size_t width = config_.encoder.hidden;
  const std::size_t rows = patches * (patch + 1);
  if (encoder_cache_.capacity() < rows)
    encoder_cache_ = KvCache(config_.encoder, std::max<std::size_t>(rows, 64));
  if (encoder_rows_count_ != patches) {
    std::vector<std::int32_t> positions(rows);
    std::vector<std::int32_t> firsts(patches);
    for (std::size_t row = 0; row < rows; ++row)
      positions[row] = static_cast<std::int32_t>(row % (patch + 1));
    for (std::size_t index = 0; index < patches; ++index)
      firsts[index] = static_cast<std::int32_t>(index * (patch + 1));
    Upload(local_positions_, positions);
    Upload(local_rows_, Iota(rows));
    Upload(local_firsts_, firsts);
    encoder_rows_count_ = patches;
  }
  encoder_bf16_.Reserve(rows * std::max(width, config_.feat_dim));
  encoder_a_.Reserve(rows * width);
  encoder_b_.Reserve(rows * width);
  encoder_c_.Reserve(rows * width);
  kernels::ToBf16(latents, encoder_bf16_.get(),
                  patches * patch * config_.feat_dim, false, stream_);
  Project(w_->encoder_in, encoder_bf16_.get(), encoder_a_.get(),
          patches * patch, *gemm_, stream_);
  kernels::PrependSpecial(encoder_a_.get(), w_->special, encoder_b_.get(),
                          patches, patch, width, stream_);
  AttentionLayout layout;
  layout.positions = local_positions_.get();
  layout.cache_rows = local_rows_.get();
  layout.rows_per_sequence = patch + 1;
  layout.keys_per_sequence = patch + 1;
  layout.causal = false;
  ForwardStack(w_->encoder, rope_.get(), layout, encoder_cache_,
               encoder_b_.get(), encoder_c_.get(), rows, scratch_, *gemm_,
               stream_);
  kernels::GatherRows(encoder_c_.get(), local_firsts_.get(), encoder_a_.get(),
                      patches, width, stream_);
  kernels::ToBf16(encoder_a_.get(), encoder_bf16_.get(), patches * width, false,
                  stream_);
  Project(w_->enc_to_lm, encoder_bf16_.get(), embeds, patches, *gemm_, stream_);
}

void Generator::EvaluateDit(const float* x, float t, float* velocity) {
  const std::size_t patch = config_.patch_size;
  const std::size_t feat = config_.feat_dim;
  const std::size_t width = config_.dit.hidden;
  const std::size_t rows = 2 * (3 + 2 * patch);
  const float* time = TimeVector(t);
  dit_bf16_.Reserve(rows * width);
  x_rows_.Reserve(patch * width);
  dit_hidden_.Reserve(rows * width);
  dit_output_.Reserve(rows * width);
  kernels::ToBf16(x, dit_bf16_.get(), patch * feat, false, stream_);
  Project(w_->dit_in, dit_bf16_.get(), x_rows_.get(), patch, *gemm_, stream_);
  kernels::AssembleDit(mu_.get(), time, cond_rows_.get(), x_rows_.get(),
                       dit_hidden_.get(), patch, width, stream_);
  AttentionLayout layout;
  layout.positions = dit_positions_.get();
  layout.cache_rows = dit_rows_.get();
  layout.rows_per_sequence = rows / 2;
  layout.keys_per_sequence = rows / 2;
  layout.causal = false;
  ForwardStack(w_->dit, rope_.get(), layout, dit_cache_, dit_hidden_.get(),
               dit_output_.get(), rows, scratch_, *gemm_, stream_);
  kernels::GatherRows(dit_output_.get(), dit_outputs_.get(), dit_hidden_.get(),
                      2 * patch, width, stream_);
  kernels::ToBf16(dit_hidden_.get(), dit_bf16_.get(), 2 * patch * width, false,
                  stream_);
  Project(w_->dit_out, dit_bf16_.get(), velocity, 2 * patch, *gemm_, stream_);
}

void Generator::Prefill(const Prompt& prompt) {
  const std::size_t rows = prompt.tokens.size();
  const std::size_t hidden = config_.lm.hidden;
  const std::size_t patch_values = config_.patch_size * config_.feat_dim;
  if (rows == 0 || rows >= capacity() || prompt.text_mask.size() != rows ||
      prompt.audio_mask.size() != rows ||
      prompt.feats.size() != rows * patch_values)
    throw std::invalid_argument("malformed VoxCPM2 prompt");
  Upload(ids_, prompt.tokens);
  Upload(text_mask_, prompt.text_mask);
  Upload(audio_mask_, prompt.audio_mask);
  prefill_a_.Reserve(rows * hidden);
  prefill_b_.Reserve(rows * hidden);
  prefill_c_.Reserve(rows * hidden);
  prefill_d_.Reserve(rows * hidden);
  prefill_bf16_.Reserve(rows * 2 * hidden);
  float* combined = prefill_a_.get();
  float* audio_embeds = prefill_b_.get();
  kernels::Embed(w_->embed, ids_.get(), text_mask_.get(), combined, rows,
                 hidden, 1.0F, stream_);
  Check(hipMemsetAsync(audio_embeds, 0, rows * hidden * sizeof(float), stream_),
        "hipMemset");

  std::vector<std::int32_t> audio_rows;
  std::vector<float> latents;
  for (std::size_t row = 0; row < rows; ++row) {
    if (prompt.audio_mask[row] == 0)
      continue;
    audio_rows.push_back(static_cast<std::int32_t>(row));
    latents.insert(
        latents.end(),
        prompt.feats.begin() + static_cast<std::ptrdiff_t>(row * patch_values),
        prompt.feats.begin() +
            static_cast<std::ptrdiff_t>((row + 1) * patch_values));
  }
  if (!audio_rows.empty()) {
    const std::size_t count = audio_rows.size();
    Upload(audio_rows_, audio_rows);
    prompt_latents_.Reserve(latents.size());
    prompt_embeds_.Reserve(count * hidden);
    Check(hipMemcpyAsync(prompt_latents_.get(), latents.data(),
                         latents.size() * sizeof(float), hipMemcpyHostToDevice,
                         stream_),
          "hipMemcpy latents");
    Encode(prompt_latents_.get(), count, prompt_embeds_.get());
    kernels::ScatterRows(prompt_embeds_.get(), audio_rows_.get(), combined,
                         count, hidden, stream_);
    kernels::ScatterRows(prompt_embeds_.get(), audio_rows_.get(), audio_embeds,
                         count, hidden, stream_);
    Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  }

  const std::vector<std::int32_t> positions = Iota(rows);
  Upload(positions_, positions);
  Upload(cache_rows_, positions);
  AttentionLayout layout;
  layout.positions = positions_.get();
  layout.cache_rows = cache_rows_.get();
  layout.rows_per_sequence = rows;
  layout.causal = true;
  float* lm_output = prefill_c_.get();
  ForwardStack(w_->base_lm, rope_.get(), layout, base_cache_, combined,
               lm_output, rows, scratch_, *gemm_, stream_);
  if (trace_ != nullptr)
    trace_->lm_prefill = Download(lm_output, rows * hidden);
  float* quantized = prefill_a_.get();  // the combined rows are spent
  Fsq(lm_output, quantized, rows);
  float* selected = prefill_d_.get();
  kernels::SelectRows(quantized, lm_output, audio_mask_.get(), selected, rows,
                      hidden, stream_);
  Check(
      hipMemcpyAsync(lm_hidden_.get(), selected + (rows - 1) * hidden,
                     hidden * sizeof(float), hipMemcpyDeviceToDevice, stream_),
      "hipMemcpy lm hidden");

  kernels::ConcatRows(selected, nullptr, hidden, audio_embeds, nullptr, hidden,
                      prefill_bf16_.get(), rows, stream_);
  float* residual_input = prefill_a_.get();
  Project(w_->fusion, prefill_bf16_.get(), residual_input, rows, *gemm_,
          stream_);
  float* residual_output = prefill_c_.get();
  ForwardStack(w_->residual_lm, nullptr, layout, residual_cache_,
               residual_input, residual_output, rows, scratch_, *gemm_,
               stream_);
  if (trace_ != nullptr)
    trace_->residual_prefill = Download(residual_output, rows * hidden);
  Check(hipMemcpyAsync(
            residual_hidden_.get(), residual_output + (rows - 1) * hidden,
            hidden * sizeof(float), hipMemcpyDeviceToDevice, stream_),
        "hipMemcpy residual hidden");
  Check(hipMemcpyAsync(
            prefix_cond_.get(), prompt.feats.data() + (rows - 1) * patch_values,
            patch_values * sizeof(float), hipMemcpyHostToDevice, stream_),
        "hipMemcpy prefix");
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  position_ = rows;
}

void Generator::Step(const float* noise, std::size_t timesteps, float cfg,
                     float* latent, float* stop_logits) {
  if (position_ == 0 || position_ >= capacity())
    throw std::logic_error("VoxCPM2 generator has no room for a patch");
  const std::size_t hidden = config_.lm.hidden;
  const std::size_t dit = config_.dit.hidden;
  const std::size_t patch_values = config_.patch_size * config_.feat_dim;
  step_bf16_.Reserve(2 * hidden);
  step_a_.Reserve(hidden);
  step_b_.Reserve(hidden);
  embed_.Reserve(hidden);

  // DiT condition and the stop predictor read the state before the patch.
  kernels::ToBf16(lm_hidden_.get(), step_bf16_.get(), hidden, false, stream_);
  Project(w_->lm_to_dit, step_bf16_.get(), mu_.get(), 1, *gemm_, stream_);
  Project(w_->stop_proj, step_bf16_.get(), step_a_.get(), 1, *gemm_, stream_);
  kernels::ToBf16(step_a_.get(), step_bf16_.get(), hidden, true, stream_);
  Project(w_->stop_head, step_bf16_.get(), stop_.get(), 1, *gemm_, stream_);
  kernels::ToBf16(residual_hidden_.get(), step_bf16_.get(), hidden, false,
                  stream_);
  Project(w_->res_to_dit, step_bf16_.get(), mu_.get() + dit, 1, *gemm_,
          stream_);
  if (trace_ != nullptr)
    trace_->mu.push_back(Download(mu_.get(), 2 * dit));
  kernels::ToBf16(prefix_cond_.get(), step_bf16_.get(), patch_values, false,
                  stream_);
  Project(w_->dit_cond, step_bf16_.get(), cond_rows_.get(), config_.patch_size,
          *gemm_, stream_);

  std::copy(noise, noise + patch_values, pinned_noise_);
  Check(hipMemcpyAsync(x_.get(), pinned_noise_, patch_values * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy noise");
  for (const EulerStep& step : EulerSchedule(timesteps)) {
    if (!step.evaluate)
      continue;  // zero velocity
    EvaluateDit(x_.get(), step.t, velocity_.get());
    kernels::GuidedEuler(velocity_.get(), x_.get(), patch_values, cfg, step.dt,
                         stream_);
  }
  float* slot = latent_ring_.get() + (position_ % 2) * patch_values;
  Check(hipMemcpyAsync(slot, x_.get(), patch_values * sizeof(float),
                       hipMemcpyDeviceToDevice, stream_),
        "hipMemcpy latent");
  Check(hipEventRecord(latent_event_, stream_), "hipEventRecord");
  latent_slot_ = slot;
  if (latent != nullptr)
    Check(hipMemcpyAsync(latent, x_.get(), patch_values * sizeof(float),
                         hipMemcpyDeviceToHost, stream_),
          "hipMemcpy latent");
  Check(
      hipMemcpyAsync(prefix_cond_.get(), x_.get(), patch_values * sizeof(float),
                     hipMemcpyDeviceToDevice, stream_),
      "hipMemcpy prefix");
  Check(hipMemcpyAsync(stop_logits, stop_.get(), 2 * sizeof(float),
                       hipMemcpyDeviceToHost, stream_),
        "hipMemcpy stop");

  // The patch's embedding advances both LMs.
  Encode(x_.get(), 1, embed_.get());
  // One row at the next position: its rotary row and cache row coincide.
  *pinned_position_ = static_cast<std::int32_t>(position_);
  positions_.Reserve(1);
  Check(hipMemcpyAsync(positions_.get(), pinned_position_, sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy position");
  AttentionLayout layout;
  layout.positions = positions_.get();
  layout.cache_rows = positions_.get();
  layout.rows_per_sequence = 1;
  layout.past = position_;
  layout.causal = true;
  Check(hipMemcpyAsync(step_a_.get(), embed_.get(), hidden * sizeof(float),
                       hipMemcpyDeviceToDevice, stream_),
        "hipMemcpy embed");
  ForwardStack(w_->base_lm, rope_.get(), layout, base_cache_, step_a_.get(),
               step_b_.get(), 1, scratch_, *gemm_, stream_);
  if (trace_ != nullptr)
    trace_->lm_step.push_back(Download(step_b_.get(), hidden));
  Fsq(step_b_.get(), lm_hidden_.get(), 1);
  kernels::ConcatRows(lm_hidden_.get(), nullptr, hidden, embed_.get(), nullptr,
                      hidden, step_bf16_.get(), 1, stream_);
  Project(w_->fusion, step_bf16_.get(), step_a_.get(), 1, *gemm_, stream_);
  ForwardStack(w_->residual_lm, nullptr, layout, residual_cache_, step_a_.get(),
               residual_hidden_.get(), 1, scratch_, *gemm_, stream_);
  if (trace_ != nullptr)
    trace_->residual_step.push_back(Download(residual_hidden_.get(), hidden));
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  ++position_;
}

void Generator::Velocity(const float* mu, const float* cond, const float* x,
                         float t, float* velocity) {
  const std::size_t patch_values = config_.patch_size * config_.feat_dim;
  Check(hipMemcpyAsync(mu_.get(), mu, 2 * config_.dit.hidden * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy mu");
  Check(hipMemcpyAsync(prefix_cond_.get(), cond, patch_values * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy cond");
  Check(hipMemcpyAsync(x_.get(), x, patch_values * sizeof(float),
                       hipMemcpyHostToDevice, stream_),
        "hipMemcpy x");
  step_bf16_.Reserve(patch_values);
  kernels::ToBf16(prefix_cond_.get(), step_bf16_.get(), patch_values, false,
                  stream_);
  Project(w_->dit_cond, step_bf16_.get(), cond_rows_.get(), config_.patch_size,
          *gemm_, stream_);
  EvaluateDit(x_.get(), t, velocity_.get());
  Check(hipMemcpyAsync(velocity, velocity_.get(),
                       2 * patch_values * sizeof(float), hipMemcpyDeviceToHost,
                       stream_),
        "hipMemcpy velocity");
  Check(hipStreamSynchronize(stream_), "hipStreamSynchronize");
}

}  // namespace gufo::models::voxcpm
