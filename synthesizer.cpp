// Follows OpenBMB/VoxCPM's VoxCPM2Model._generate
// (src/voxcpm/model/voxcpm2.py; Apache-2.0, Copyright OpenBMB; see
// licenses/VoxCPM.txt): the prompt of each mode, the stop rules and the retry
// of overlong results. Rewritten in C++ for Gufo.

#include "synthesizer.hpp"

#include <algorithm>
#include <gufo/ext/audio.hpp>
#include <random>
#include <stdexcept>
#include <string_view>

namespace gufo::models::voxcpm {
namespace {

// VoxCPM2Model's control tokens.
constexpr std::int32_t kAudioStart = 101;
constexpr std::int32_t kReferenceStart = 103;
constexpr std::int32_t kReferenceEnd = 104;
// _inference: no stop before the fourth patch.
constexpr std::size_t kMinPatches = 2;
// retry_badcase_ratio_threshold and the cap max_len = 6 * tokens + 10.
constexpr double kPatchesPerToken = 6.0;
constexpr std::size_t kRetries = 3;
// Continuations decode the last (streaming_prefix_len - 1) prompt patches
// first, as the reference does before trimming them.
constexpr std::size_t kContextPatches = 3;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

/// Unicode whitespace as Python's str regex \s matches it, at `text[i]`;
/// returns its byte length or 0.
std::size_t WhitespaceAt(std::string_view text, std::size_t i) {
  const auto byte = static_cast<unsigned char>(text[i]);
  if (byte == ' ' || (byte >= 0x09 && byte <= 0x0D) ||
      (byte >= 0x1C && byte <= 0x1F))
    return 1;
  const auto starts = [&](std::string_view sequence) {
    return text.substr(i, sequence.size()) == sequence;
  };
  if (starts("\xC2\x85") || starts("\xC2\xA0"))
    return 2;
  if (starts("\xE1\x9A\x80") || starts("\xE2\x80\xA8") ||
      starts("\xE2\x80\xA9") || starts("\xE2\x80\xAF") ||
      starts("\xE2\x81\x9F") || starts("\xE3\x80\x80"))
    return 3;
  if (byte == 0xE2 && i + 2 < text.size() &&
      static_cast<unsigned char>(text[i + 1]) == 0x80 &&
      static_cast<unsigned char>(text[i + 2]) <= 0x8A)
    return 3;  // U+2000..U+200A
  return 0;
}

struct Built {
  Prompt prompt;
  std::size_t target_tokens = 0;
  // Prompt patches the decoder sees before the generated ones.
  std::vector<float> context;
};

}  // namespace

std::string CollapseWhitespace(const std::string& text) {
  std::string result;
  bool space = false;
  for (std::size_t i = 0; i < text.size();) {
    const std::size_t length = WhitespaceAt(text, i);
    if (length > 0) {
      space = true;
      i += length;
      continue;
    }
    if (space)
      result.push_back(' ');
    space = false;
    result.push_back(text[i++]);
  }
  if (space)
    result.push_back(' ');
  return result;
}

std::unique_ptr<Synthesizer> Synthesizer::Load(
    const std::filesystem::path& model_root, std::string* error) {
  std::unique_ptr<Synthesizer> synthesizer(new Synthesizer());
  // The files stay mapped until the weights are on the device.
  const auto files = ModelFiles::Open(model_root, error);
  if (files == nullptr)
    return nullptr;
  const auto config = ParseConfig(files->config(), error);
  if (!config.has_value() || !IsSupported(*config, error))
    return nullptr;
  synthesizer->config_ = *config;
  synthesizer->tokenizer_ = Tokenizer::Parse(files->tokenizer(), error);
  if (synthesizer->tokenizer_ == nullptr)
    return nullptr;
  synthesizer->generator_ = Generator::Create(*config, *files, error);
  if (synthesizer->generator_ == nullptr)
    return nullptr;
  synthesizer->vae_ = AudioVae::Create(*config, *files, error);
  if (synthesizer->vae_ == nullptr)
    return nullptr;
  try {
    synthesizer->Warmup();
  } catch (const std::exception& failure) {
    SetError(error, std::string("VoxCPM2 warm-up failed: ") + failure.what());
    return nullptr;
  }
  return synthesizer;
}

void Synthesizer::Warmup() {
  // A reference prompt of 8 patches and 64 text rows, one patch and its
  // decoding: hipBLASLt times its candidates for the prompt products (up
  // to 128 rows) and the DiT's time embeddings are computed, so that the
  // first request doesn't wait for either.
  const std::size_t patch_values = config_.patch_size * config_.feat_dim;
  Prompt prompt;
  const auto add = [&](std::int32_t token, bool text_row) {
    prompt.tokens.push_back(token);
    prompt.text_mask.push_back(text_row ? 1 : 0);
    prompt.audio_mask.push_back(text_row ? 0 : 1);
    prompt.feats.resize(prompt.feats.size() + patch_values, 0.0F);
  };
  add(kReferenceStart, true);
  for (int row = 0; row < 8; ++row)
    add(0, false);
  add(kReferenceEnd, true);
  for (int row = 0; row < 64; ++row)
    add(1000 + row, true);
  add(kAudioStart, true);
  generator_->Prefill(prompt);
  const std::vector<float> noise(patch_values, 0.0F);
  float stop[2];
  SynthesisOptions defaults;
  generator_->Step(noise.data(), defaults.timesteps, defaults.cfg, nullptr,
                   stop);
  std::vector<float> audio;
  vae_->Reset();
  vae_->Enqueue(generator_->latent(), config_.patch_size,
                generator_->latent_event());
  vae_->Collect(&audio);
  vae_->Reset();
}

bool Synthesizer::CloneVoice(std::span<const float> samples,
                             std::uint32_t channels, std::uint32_t sample_rate,
                             const std::string& transcript,
                             ReferenceVoice* voice, std::string* error) {
  try {
    const std::vector<float> mono = core::DownmixMono(samples, channels);
    const auto rate = static_cast<std::uint32_t>(config_.vae.sample_rate);
    std::vector<float> audio = core::Resample(mono, sample_rate, rate);
    if (audio.empty()) {
      SetError(error, "the reference recording is empty");
      return false;
    }
    // _encode_wav: pad to whole patches, on the right for the isolated
    // reference and on the left for a continued prompt.
    const std::size_t patch = config_.patch_size * config_.vae.hop();
    const std::size_t padding = (patch - audio.size() % patch) % patch;
    std::vector<float> right = audio;
    right.resize(audio.size() + padding, 0.0F);
    voice->reference = vae_->Encode(right);
    voice->prompt.clear();
    // The transcript is used as given, like the reference's prompt_text.
    const std::string cleaned = CollapseWhitespace(transcript);
    voice->transcript = transcript;
    if (!cleaned.empty() && cleaned != " ") {
      std::vector<float> left(padding, 0.0F);
      left.insert(left.end(), audio.begin(), audio.end());
      voice->prompt = vae_->Encode(left);
    } else {
      voice->transcript.clear();
    }
  } catch (const std::exception& failure) {
    SetError(error,
             std::string("cannot encode the reference: ") + failure.what());
    return false;
  }
  return true;
}

bool Synthesizer::Synthesize(const SynthesisOptions& options,
                             const Cancelled& cancelled, const AudioSink& sink,
                             std::vector<float>* audio, std::string* error) {
  const std::string text = CollapseWhitespace(options.text);
  if (text.empty() || text == " ") {
    SetError(error, "the text to synthesize is empty");
    return false;
  }
  const std::string target =
      options.instruct.empty() ? text : "(" + options.instruct + ")" + text;
  const ReferenceVoice* voice = options.voice;
  const bool continuation = voice != nullptr && !voice->prompt.empty();
  const std::size_t patch_values = config_.patch_size * config_.feat_dim;

  Built built;
  std::vector<std::uint32_t> ids;
  std::vector<std::uint32_t> target_ids;
  if (!tokenizer_->Encode(continuation ? voice->transcript + target : target,
                          &ids, error) ||
      !tokenizer_->Encode(target, &target_ids, error))
    return false;
  built.target_tokens = target_ids.size();
  Prompt& prompt = built.prompt;
  const auto add = [&](std::int32_t token, bool text_row, const float* feat) {
    prompt.tokens.push_back(token);
    prompt.text_mask.push_back(text_row ? 1 : 0);
    prompt.audio_mask.push_back(text_row ? 0 : 1);
    if (feat == nullptr)
      prompt.feats.resize(prompt.feats.size() + patch_values, 0.0F);
    else
      prompt.feats.insert(prompt.feats.end(), feat, feat + patch_values);
  };
  if (voice != nullptr) {
    add(kReferenceStart, true, nullptr);
    for (std::size_t offset = 0; offset < voice->reference.size();
         offset += patch_values)
      add(0, false, voice->reference.data() + offset);
    add(kReferenceEnd, true, nullptr);
  }
  for (const std::uint32_t id : ids)
    add(static_cast<std::int32_t>(id), true, nullptr);
  add(kAudioStart, true, nullptr);
  if (continuation) {
    for (std::size_t offset = 0; offset < voice->prompt.size();
         offset += patch_values)
      add(0, false, voice->prompt.data() + offset);
    const std::size_t patches = voice->prompt.size() / patch_values;
    const std::size_t context = std::min(kContextPatches, patches);
    built.context.assign(voice->prompt.end() - static_cast<std::ptrdiff_t>(
                                                   context * patch_values),
                         voice->prompt.end());
  }
  const std::size_t rows = prompt.tokens.size();
  if (rows + 1 >= generator_->capacity()) {
    SetError(error, "the text and reference exceed VoxCPM2's " +
                        std::to_string(generator_->capacity()) + " positions");
    return false;
  }
  std::size_t limit = std::min(
      options.max_patches,
      static_cast<std::size_t>(
          static_cast<double>(built.target_tokens) * kPatchesPerToken + 10.0));
  limit = std::min(limit, generator_->capacity() - rows);

  try {
    std::uint64_t seed = options.seed;
    for (std::size_t attempt = 0;; ++attempt) {
      std::mt19937_64 random(seed);
      std::normal_distribution<float> normal(0.0F, 1.0F);
      std::vector<float> noise(patch_values);
      std::vector<float> decoded;
      float stop[2] = {0.0F, 0.0F};
      if (audio != nullptr)
        audio->clear();
      generator_->Prefill(prompt);
      vae_->Reset();
      if (!built.context.empty())
        vae_->Decode(built.context.data(),
                     built.context.size() / config_.feat_dim, &decoded);
      std::size_t patches = 0;
      // A patch decodes on the AudioVAE's stream while the next one is
      // generated; `deliver` hands the decoded patch on.
      bool pending = false;
      const auto deliver = [&]() {
        if (!pending)
          return true;
        pending = false;
        decoded.clear();
        vae_->Collect(&decoded);
        if (sink)
          return sink(decoded);
        if (audio != nullptr)
          audio->insert(audio->end(), decoded.begin(), decoded.end());
        return true;
      };
      for (std::size_t index = 0; index < limit; ++index) {
        if (cancelled && cancelled()) {
          SetError(error, "cancelled");
          return false;
        }
        for (float& value : noise)
          value = normal(random);
        generator_->Step(noise.data(), options.timesteps, options.cfg, nullptr,
                         stop);
        ++patches;
        if (!deliver()) {
          SetError(error, "cancelled");
          return false;
        }
        vae_->Enqueue(generator_->latent(), config_.patch_size,
                      generator_->latent_event());
        pending = true;
        // The first patch is sent at once, for the first audio's latency.
        if (index == 0 && !deliver()) {
          SetError(error, "cancelled");
          return false;
        }
        if (index > kMinPatches && stop[1] > stop[0])
          break;
      }
      if (!deliver()) {
        SetError(error, "cancelled");
        return false;
      }
      const bool runaway =
          static_cast<double>(patches) >=
          static_cast<double>(built.target_tokens) * kPatchesPerToken;
      if (!runaway || sink || !options.retry || attempt + 1 >= kRetries)
        break;
      ++seed;
    }
  } catch (const std::exception& failure) {
    SetError(error, std::string("VoxCPM2 synthesis failed: ") + failure.what());
    return false;
  }
  return true;
}

}  // namespace gufo::models::voxcpm
