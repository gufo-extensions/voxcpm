#include "runner.hpp"

#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "synthesizer.hpp"

namespace gufo::models::voxcpm {
namespace {

constexpr const char* kAutoVoice = "auto";
constexpr const char* kDesignVoice = "voice-design";
// Encoded reference voices kept for repeated requests (presets reuse theirs).
constexpr std::size_t kCachedVoices = 8;

std::uint64_t Fingerprint(const ext::TtsAudioBuffer& audio,
                          const std::string& text) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&hash](const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
      hash ^= p[i];
      hash *= 1099511628211ULL;
    }
  };
  mix(&audio.sample_rate, sizeof(audio.sample_rate));
  mix(&audio.channels, sizeof(audio.channels));
  mix(audio.samples.data(), audio.samples.size() * sizeof(float));
  mix(text.data(), text.size());
  return hash;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

class VoxCpm2Runner final : public ext::TtsRunner {
public:
  explicit VoxCpm2Runner(std::unique_ptr<Synthesizer> synthesizer)
      : synthesizer_(std::move(synthesizer)) {}

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    if (request.speaker_embedding_only) {
      SetError(error,
               "VoxCPM2 clones a voice from its recording; send the reference "
               "audio instead of speaker_embedding_only");
      return false;
    }
    SynthesisOptions options;
    options.text = request.text;
    options.instruct = request.instruct;
    options.max_patches = request.max_new_tokens;
    options.seed = request.sampling.seed;
    if (!request.reference_audio.samples.empty()) {
      options.voice = Reference(request, error);
      if (options.voice == nullptr)
        return false;
    }
    std::vector<float> audio;
    const bool streaming = static_cast<bool>(request.on_audio);
    std::size_t streamed = 0;
    Synthesizer::AudioSink sink;
    if (streaming) {
      sink = [&](std::span<const float> samples) {
        streamed += samples.size();
        return request.on_audio(samples);
      };
    }
    const auto cancelled = [&] { return is_cancelled && is_cancelled(); };
    if (!synthesizer_->Synthesize(options, cancelled, sink, &audio, error))
      return false;
    result->sample_rate = SampleRate();
    result->code_groups = 0;
    result->sample_count = streaming ? streamed : audio.size();
    if (!streaming)
      result->samples = std::move(audio);
    return true;
  }

  std::vector<std::string> Voices() const override {
    return {kAutoVoice, kDesignVoice};
  }

  ext::TtsVariant Variant() const override { return ext::TtsVariant::kBase; }

  std::uint32_t SampleRate() const override {
    return synthesizer_->SampleRate();
  }

  ext::TtsConditioningCapabilities Conditioning() const override {
    // One checkpoint serves an automatic voice, voice design from
    // instructions and cloning from a recording or a preset, with or
    // without its transcript.
    ext::TtsConditioningCapabilities capabilities;
    capabilities.unconditioned = true;
    capabilities.require_voice = false;
    capabilities.default_voice = kAutoVoice;
    capabilities.reference_audio = ext::TtsConditioningMode::kOptional;
    capabilities.reference_presets = true;
    capabilities.instructions = ext::TtsConditioningMode::kOptional;
    return capabilities;
  }

  std::string ModelId() const override { return "voxcpm2"; }

private:
  struct CachedVoice {
    std::uint64_t key = 0;
    std::shared_ptr<ReferenceVoice> voice;
  };

  /// The request's reference voice, encoded once per recording and
  /// transcript.
  const ReferenceVoice* Reference(const ext::SynthesisRequest& request,
                                  std::string* error) {
    const std::uint64_t key =
        Fingerprint(request.reference_audio, request.reference_text);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      if (it->key == key) {
        CachedVoice hit = *it;
        cache_.erase(it);
        cache_.push_front(hit);
        return cache_.front().voice.get();
      }
    }
    auto voice = std::make_shared<ReferenceVoice>();
    const ext::TtsAudioBuffer& audio = request.reference_audio;
    if (!synthesizer_->CloneVoice(audio.samples, audio.channels,
                                  audio.sample_rate, request.reference_text,
                                  voice.get(), error))
      return nullptr;
    cache_.push_front({key, std::move(voice)});
    if (cache_.size() > kCachedVoices)
      cache_.pop_back();
    return cache_.front().voice.get();
  }

  std::unique_ptr<Synthesizer> synthesizer_;
  std::deque<CachedVoice> cache_;
};

}  // namespace

std::unique_ptr<ext::TtsRunner> CreateVoxCpm2Runner(
    const ext::TtsRunnerOptions& options, std::string* error) {
  auto synthesizer = Synthesizer::Load(options.model_root, error);
  if (synthesizer == nullptr)
    return nullptr;
  return std::make_unique<VoxCpm2Runner>(std::move(synthesizer));
}

}  // namespace gufo::models::voxcpm
