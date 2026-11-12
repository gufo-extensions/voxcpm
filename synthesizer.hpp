#ifndef GUFO_MODELS_VOXCPM_SYNTHESIZER_HPP_
#define GUFO_MODELS_VOXCPM_SYNTHESIZER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "config.hpp"
#include "generator.hpp"
#include "tokenizer.hpp"
#include "vae.hpp"

namespace gufo::models::voxcpm {

/// A voice to clone, encoded once (VoxCPM2Model.build_prompt_cache):
/// the recording as isolated reference latents and, with its transcript,
/// also as the prompt that generation continues.
struct ReferenceVoice {
  /// [patches][patch_size][feat_dim]; the recording padded on the right.
  std::vector<float> reference;
  /// [patches][patch_size][feat_dim]; padded on the left. Empty without a
  /// transcript.
  std::vector<float> prompt;
  std::string transcript;
};

struct SynthesisOptions {
  std::string text;
  /// Voice design or style hint; prepended in parentheses.
  std::string instruct;
  const ReferenceVoice* voice = nullptr;
  /// Upper bound of generated patches; the reference also stops at
  /// 6 patches per text token plus 10.
  std::size_t max_patches = 4096;
  std::uint64_t seed = 0;
  float cfg = 2.0F;
  std::size_t timesteps = 10;
  /// Retries (seed + 1, at most three attempts) a buffered result that ran
  /// into the text-length cap, as the reference's retry_badcase does.
  bool retry = true;
};

/// Text to 48 kHz speech with VoxCPM2: prompt construction for the
/// zero-shot, reference, continuation and combined modes, the patch loop
/// with its stop predictor, and AudioVAE decoding patch by patch.
class Synthesizer {
public:
  using Cancelled = std::function<bool()>;
  /// Receives each decoded patch; returning false cancels.
  using AudioSink = std::function<bool(std::span<const float>)>;

  [[nodiscard]] static std::unique_ptr<Synthesizer> Load(
      const std::filesystem::path& model_root, std::string* error = nullptr);

  [[nodiscard]] std::uint32_t SampleRate() const {
    return static_cast<std::uint32_t>(vae_->output_rate());
  }
  [[nodiscard]] const Config& config() const { return config_; }

  /// Encodes interleaved audio at any rate and its optional transcript.
  [[nodiscard]] bool CloneVoice(std::span<const float> samples,
                                std::uint32_t channels,
                                std::uint32_t sample_rate,
                                const std::string& transcript,
                                ReferenceVoice* voice,
                                std::string* error = nullptr);

  /// Synthesizes `options.text`. With `sink`, audio is delivered patch by
  /// patch and not kept; otherwise it is returned in `audio`.
  [[nodiscard]] bool Synthesize(const SynthesisOptions& options,
                                const Cancelled& cancelled,
                                const AudioSink& sink,
                                std::vector<float>* audio,
                                std::string* error = nullptr);

private:
  Synthesizer() = default;
  void Warmup();

  Config config_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Generator> generator_;
  std::unique_ptr<AudioVae> vae_;
};

/// The reference's text cleanup: newlines become spaces and whitespace runs
/// one space.
[[nodiscard]] std::string CollapseWhitespace(const std::string& text);

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_SYNTHESIZER_HPP_
