#include "config.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::voxcpm {
namespace {

namespace json = gufo::json;

class ConfigError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Typed access to one JSON object; failures name the key. Optional keys
/// take the reference's pydantic defaults.
class Fields {
public:
  Fields(const json::Value& object, std::string where)
      : object_(object), where_(std::move(where)) {
    if (!object_.is_object())
      throw ConfigError(where_ + " is not a JSON object");
  }

  [[nodiscard]] const json::Value* Find(const std::string& key) const {
    const json::Value* value = object_.find(key);
    return value == nullptr || value->is_null() ? nullptr : value;
  }

  [[nodiscard]] const json::Value& Get(const std::string& key) const {
    const json::Value* value = Find(key);
    if (value == nullptr)
      throw ConfigError(where_ + " lacks \"" + key + "\"");
    return *value;
  }

  [[nodiscard]] std::size_t Size(const std::string& key) const {
    const std::size_t size = Get(key).as_size(0);
    if (size == 0)
      throw ConfigError(where_ + "." + key + " is not a positive integer");
    return size;
  }

  [[nodiscard]] std::size_t Size(const std::string& key,
                                 std::size_t fallback) const {
    return Find(key) == nullptr ? fallback : Size(key);
  }

  [[nodiscard]] double Number(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_number() || !std::isfinite(value.as_double()))
      throw ConfigError(where_ + "." + key + " is not a number");
    return value.as_double();
  }

  [[nodiscard]] double Number(const std::string& key, double fallback) const {
    return Find(key) == nullptr ? fallback : Number(key);
  }

  [[nodiscard]] std::string String(const std::string& key,
                                   std::string fallback) const {
    const json::Value* value = Find(key);
    if (value == nullptr)
      return fallback;
    if (!value->is_string())
      throw ConfigError(where_ + "." + key + " is not a string");
    return value->str();
  }

  [[nodiscard]] bool Bool(const std::string& key, bool fallback) const {
    const json::Value* value = Find(key);
    if (value == nullptr)
      return fallback;
    if (!value->is_bool())
      throw ConfigError(where_ + "." + key + " is not a boolean");
    return value->as_bool();
  }

  [[nodiscard]] std::vector<std::size_t> Sizes(
      const std::string& key, std::vector<std::size_t> fallback) const {
    const json::Value* value = Find(key);
    if (value == nullptr)
      return fallback;
    if (!value->is_array() || value->empty())
      throw ConfigError(where_ + "." + key + " is not a non-empty array");
    std::vector<std::size_t> sizes;
    for (const json::Value& item : value->items()) {
      const std::size_t size = item.as_size(0);
      if (size == 0)
        throw ConfigError(where_ + "." + key + " holds a non-positive entry");
      sizes.push_back(size);
    }
    return sizes;
  }

  [[nodiscard]] std::vector<double> Numbers(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_array() || value.empty())
      throw ConfigError(where_ + "." + key + " is not a non-empty array");
    std::vector<double> numbers;
    for (const json::Value& item : value.items()) {
      if (!item.is_number() || !std::isfinite(item.as_double()))
        throw ConfigError(where_ + "." + key + " holds a non-number");
      numbers.push_back(item.as_double());
    }
    return numbers;
  }

  [[nodiscard]] Fields Object(const std::string& key) const {
    return Fields(Get(key), where_ + "." + key);
  }

  [[nodiscard]] bool Has(const std::string& key) const {
    return Find(key) != nullptr;
  }

private:
  const json::Value& object_;
  std::string where_;
};

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw ConfigError("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

/// The shape of a derived stack: VoxCPMEncoderConfig / VoxCPMDitConfig
/// override hidden size, FFN size, heads, layers and kv_channels of a copy
/// of lm_config (kv_channels None falls back to hidden / heads).
TransformerConfig DeriveStack(const TransformerConfig& lm,
                              const Fields& fields) {
  TransformerConfig stack = lm;
  stack.hidden = fields.Size("hidden_dim", 1024);
  stack.intermediate = fields.Size("ffn_dim", 4096);
  stack.heads = fields.Size("num_heads", 16);
  stack.layers = fields.Size("num_layers", 4);
  stack.head_dim = fields.Size("kv_channels", stack.hidden / stack.heads);
  stack.vocab = 0;
  return stack;
}

Config Parse(std::string_view text) {
  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& failure) {
    throw ConfigError(std::string("config.json: ") + failure.what());
  }
  const Fields config(root, "config.json");
  const std::string architecture = config.String("architecture", "");
  if (architecture != "voxcpm2")
    throw ConfigError("config.json names architecture \"" + architecture +
                      "\", expected \"voxcpm2\"");

  Config result;
  const Fields lm = config.Object("lm_config");
  TransformerConfig& base = result.lm;
  base.hidden = lm.Size("hidden_size");
  base.intermediate = lm.Size("intermediate_size");
  base.layers = lm.Size("num_hidden_layers");
  base.heads = lm.Size("num_attention_heads");
  base.kv_heads = lm.Size("num_key_value_heads");
  base.head_dim = lm.Size("kv_channels", base.hidden / base.heads);
  base.vocab = lm.Size("vocab_size");
  base.rms_eps = static_cast<float>(lm.Number("rms_norm_eps"));
  base.rope = !lm.Bool("no_rope", false);
  base.use_mup = lm.Bool("use_mup", true);
  base.scale_emb = static_cast<float>(lm.Number("scale_emb"));
  base.scale_depth = static_cast<float>(lm.Number("scale_depth"));

  const Fields scaling = lm.Object("rope_scaling");
  result.rope.theta = lm.Number("rope_theta");
  result.rope.dim = base.head_dim;
  result.rope.max_positions = lm.Size("max_position_embeddings");
  result.rope.original_max_positions =
      scaling.Size("original_max_position_embeddings");
  result.rope.short_factor = scaling.Numbers("short_factor");
  result.rope.long_factor = scaling.Numbers("long_factor");

  result.residual_lm = base;
  result.residual_lm.layers = config.Size("residual_lm_num_layers", 8);
  result.residual_lm.vocab = 0;
  result.residual_lm.rope = !config.Bool("residual_lm_no_rope", false);
  result.encoder = DeriveStack(base, config.Object("encoder_config"));
  const Fields dit = config.Object("dit_config");
  result.dit = DeriveStack(base, dit);
  if (dit.Bool("dit_mean_mode", false))
    throw ConfigError("config.json: dit_mean_mode is not supported");

  result.patch_size = config.Size("patch_size", 4);
  result.feat_dim = config.Size("feat_dim", 64);
  result.fsq_latent_dim = config.Size("scalar_quantization_latent_dim", 512);
  result.fsq_scale =
      static_cast<float>(config.Number("scalar_quantization_scale", 9));
  result.max_length = config.Size("max_length", 8192);
  result.dtype = config.String("dtype", "bfloat16");

  // AudioVAEConfigV2's defaults apply to absent keys.
  AudioVaeConfig& vae = result.vae;
  vae.encoder_dim = 128;
  vae.encoder_rates = {2, 5, 8, 8};
  vae.latent_dim = 64;
  vae.decoder_dim = 2048;
  vae.decoder_rates = {8, 6, 5, 2, 2, 2};
  vae.sample_rate = 16000;
  vae.out_sample_rate = 48000;
  vae.sr_bin_boundaries = {20000, 30000, 40000};
  if (config.Has("audio_vae_config")) {
    const Fields fields = config.Object("audio_vae_config");
    vae.encoder_dim = fields.Size("encoder_dim", vae.encoder_dim);
    vae.encoder_rates = fields.Sizes("encoder_rates", vae.encoder_rates);
    vae.latent_dim = fields.Size("latent_dim", vae.latent_dim);
    vae.decoder_dim = fields.Size("decoder_dim", vae.decoder_dim);
    vae.decoder_rates = fields.Sizes("decoder_rates", vae.decoder_rates);
    vae.sample_rate = fields.Size("sample_rate", vae.sample_rate);
    vae.out_sample_rate = fields.Size("out_sample_rate", vae.out_sample_rate);
    vae.sr_bin_boundaries =
        fields.Sizes("sr_bin_boundaries", vae.sr_bin_boundaries);
    if (!fields.Bool("depthwise", true) ||
        fields.Bool("use_noise_block", false) ||
        fields.String("cond_type", "scale_bias") != "scale_bias" ||
        fields.Bool("cond_out_layer", false))
      throw ConfigError(
          "config.json: only the depthwise AudioVAE V2 with scale_bias "
          "rate conditioning and no noise blocks is supported");
  }
  return result;
}

std::optional<Config> Fail(std::string* error, const std::string& message) {
  if (error != nullptr)
    *error = message;
  return std::nullopt;
}

}  // namespace

double RopeConfig::scaling() const {
  const double scale = static_cast<double>(max_positions) /
                       static_cast<double>(original_max_positions);
  return std::sqrt(1.0 + std::log(scale) / std::log(static_cast<double>(
                                               original_max_positions)));
}

std::size_t AudioVaeConfig::hop() const {
  return std::accumulate(encoder_rates.begin(), encoder_rates.end(),
                         std::size_t{1}, std::multiplies<>());
}

std::size_t AudioVaeConfig::decode_hop() const {
  return std::accumulate(decoder_rates.begin(), decoder_rates.end(),
                         std::size_t{1}, std::multiplies<>());
}

std::size_t AudioVaeConfig::rate_bucket() const {
  std::size_t bucket = 0;
  while (bucket < sr_bin_boundaries.size() &&
         sr_bin_boundaries[bucket] < out_sample_rate)
    ++bucket;
  return bucket;
}

std::optional<Config> ParseConfig(std::string_view text, std::string* error) {
  try {
    return Parse(text);
  } catch (const ConfigError& failure) {
    return Fail(error, failure.what());
  }
}

std::optional<Config> LoadConfig(std::string_view model_root,
                                 std::string* error) {
  std::string text;
  try {
    text = ReadFile(std::filesystem::path(model_root) / "config.json");
  } catch (const ConfigError& failure) {
    return Fail(error, failure.what());
  }
  return ParseConfig(text, error);
}

bool IsSupported(const Config& config, std::string* error) {
  const auto fail = [error](const std::string& message) {
    if (error != nullptr)
      *error = message;
    return false;
  };
  for (const TransformerConfig* stack :
       {&config.lm, &config.residual_lm, &config.encoder, &config.dit}) {
    if (stack->head_dim != 128)
      return fail("the attention kernels need a head size of 128");
    if (stack->heads % stack->kv_heads != 0)
      return fail("attention heads must be a multiple of the KV heads");
  }
  if (config.lm.use_mup)
    return fail("muP-scaled checkpoints are not supported");
  if (config.rope.short_factor.size() * 2 != config.rope.dim ||
      config.rope.long_factor.size() * 2 != config.rope.dim)
    return fail("the LongRoPE factors do not match the head size");
  if (config.feat_dim != config.vae.latent_dim)
    return fail("feat_dim differs from the AudioVAE latent size");
  if (config.vae.decoder_dim >> config.vae.decoder_rates.size() == 0)
    return fail("the AudioVAE decoder halves its channels below one");
  if (config.dtype != "bfloat16")
    return fail("only bfloat16 checkpoints are supported");
  return true;
}

}  // namespace gufo::models::voxcpm
