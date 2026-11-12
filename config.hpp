#ifndef GUFO_MODELS_VOXCPM_CONFIG_HPP_
#define GUFO_MODELS_VOXCPM_CONFIG_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::voxcpm {

/// One MiniCPM4 transformer stack. VoxCPM2 derives its four stacks (text LM,
/// residual LM, local encoder, local DiT) from `lm_config`, overriding the
/// sizes per stack as the reference model does.
struct TransformerConfig {
  std::size_t hidden = 0;
  std::size_t intermediate = 0;
  std::size_t layers = 0;
  std::size_t heads = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dim = 0;
  /// 0 for the stacks without a token embedding.
  std::size_t vocab = 0;
  float rms_eps = 0.0F;
  bool rope = true;
  // muP scaling: embeddings times `scale_emb`, residual branches times
  // `scale_depth / sqrt(layers)`. VoxCPM2 disables it.
  bool use_mup = false;
  float scale_emb = 1.0F;
  float scale_depth = 1.0F;
};

/// MiniCPM's LongRoPE. The reference fills its table up to
/// `max_position_embeddings` once, so the long factors apply to every
/// position when that exceeds the original length and the short ones
/// otherwise.
struct RopeConfig {
  double theta = 10000.0;
  std::size_t dim = 0;
  std::size_t max_positions = 0;
  std::size_t original_max_positions = 0;
  std::vector<double> short_factor;
  std::vector<double> long_factor;

  /// The factors the reference applies (one per frequency).
  [[nodiscard]] const std::vector<double>& factors() const {
    return max_positions > original_max_positions ? long_factor : short_factor;
  }
  /// The magnitude scale of the cos/sin table.
  [[nodiscard]] double scaling() const;
};

/// AudioVAE V2: causal DAC-style encoder at `sample_rate`, decoder to
/// `out_sample_rate` conditioned on the output rate's bucket.
struct AudioVaeConfig {
  std::size_t encoder_dim = 0;
  std::vector<std::size_t> encoder_rates;
  std::size_t latent_dim = 0;
  std::size_t decoder_dim = 0;
  std::vector<std::size_t> decoder_rates;
  std::size_t sample_rate = 0;
  std::size_t out_sample_rate = 0;
  std::vector<std::size_t> sr_bin_boundaries;

  /// Input samples per latent frame.
  [[nodiscard]] std::size_t hop() const;
  /// Output samples per latent frame.
  [[nodiscard]] std::size_t decode_hop() const;
  /// The decoder's sample-rate bucket for `out_sample_rate`
  /// (torch.bucketize, right=False).
  [[nodiscard]] std::size_t rate_bucket() const;
};

struct Config {
  TransformerConfig lm;
  TransformerConfig residual_lm;
  TransformerConfig encoder;
  TransformerConfig dit;
  RopeConfig rope;  // head_dim of every stack is lm.head_dim's
  std::size_t patch_size = 0;
  std::size_t feat_dim = 0;
  std::size_t fsq_latent_dim = 0;
  float fsq_scale = 0.0F;
  AudioVaeConfig vae;
  /// Positions in each LM's KV cache (prompt plus generated patches).
  std::size_t max_length = 0;
  std::string dtype;
};

/// Reads `config.json` of an openbmb VoxCPM2 directory.
[[nodiscard]] std::optional<Config> LoadConfig(std::string_view model_root,
                                               std::string* error = nullptr);
/// Parses the text of `config.json`.
[[nodiscard]] std::optional<Config> ParseConfig(std::string_view text,
                                                std::string* error = nullptr);
/// Whether this extension's kernels support the configuration.
[[nodiscard]] bool IsSupported(const Config& config,
                               std::string* error = nullptr);

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_CONFIG_HPP_
