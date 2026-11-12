#include "config.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "config_samples.hpp"

namespace vc = gufo::models::voxcpm;

namespace {

int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::cout << "FAIL " << what << '\n';
    ++failures;
  }
}

std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const std::size_t at = text.find(from);
  if (at == std::string::npos)
    throw std::runtime_error("sample lacks " + from);
  return text.replace(at, from.size(), to);
}

void TestReleasedConfig() {
  std::string error;
  const auto config = vc::ParseConfig(voxcpm_test::kModelConfig, &error);
  Check(config.has_value(), "released config parses: " + error);
  if (!config.has_value())
    return;
  const vc::TransformerConfig& lm = config->lm;
  Check(lm.hidden == 2048 && lm.intermediate == 6144 && lm.layers == 28 &&
            lm.heads == 16 && lm.kv_heads == 2 && lm.head_dim == 128 &&
            lm.vocab == 73448 && lm.rope && !lm.use_mup,
        "text LM shape");
  Check(std::fabs(lm.rms_eps - 1e-5F) < 1e-12F, "RMS epsilon");
  const vc::TransformerConfig& residual = config->residual_lm;
  Check(residual.hidden == 2048 && residual.layers == 8 &&
            residual.vocab == 0 && !residual.rope,
        "residual LM shape");
  for (const vc::TransformerConfig* stack : {&config->encoder, &config->dit}) {
    Check(stack->hidden == 1024 && stack->intermediate == 4096 &&
              stack->heads == 16 && stack->kv_heads == 2 &&
              stack->head_dim == 128 && stack->layers == 12 &&
              stack->vocab == 0 && stack->rope,
          "encoder and DiT shapes");
  }
  Check(config->rope.dim == 128 && config->rope.factors().size() == 64 &&
            &config->rope.factors() == &config->rope.short_factor &&
            std::fabs(config->rope.scaling() - 1.0) < 1e-12,
        "LongRoPE uses the short factors unscaled");
  Check(config->patch_size == 4 && config->feat_dim == 64 &&
            config->fsq_latent_dim == 512 && config->fsq_scale == 9.0F &&
            config->max_length == 8192,
        "patching, FSQ and cache length");
  const vc::AudioVaeConfig& vae = config->vae;
  Check(vae.hop() == 640 && vae.decode_hop() == 1920 &&
            vae.sample_rate == 16000 && vae.out_sample_rate == 48000 &&
            vae.decoder_dim == 2048 && vae.rate_bucket() == 3,
        "AudioVAE rates");
  Check(vc::IsSupported(*config, &error),
        "released config supported: " + error);
}

void TestDefaultsAndErrors() {
  const std::string sample = voxcpm_test::kModelConfig;
  std::string error;
  {
    // Without audio_vae_config, AudioVAEConfigV2's defaults apply.
    const std::size_t start = sample.find("\"audio_vae_config\"");
    const std::size_t end = sample.find('}', start);
    std::string text = sample;
    text.erase(start, end - start + 2);
    const auto config = vc::ParseConfig(text, &error);
    Check(config.has_value() && config->vae.decoder_rates.size() == 6 &&
              config->vae.rate_bucket() == 3,
          "AudioVAE defaults: " + error);
  }
  Check(!vc::ParseConfig(Replace(sample, "\"voxcpm2\"", "\"voxcpm\"")),
        "other architectures are rejected");
  Check(!vc::ParseConfig(Replace(sample, "\"lm_config\"", "\"lm\"")),
        "a missing lm_config is rejected");
  Check(!vc::ParseConfig(Replace(sample, "\"dit_config\": {",
                                 "\"dit_config\": {\"dit_mean_mode\": true,")),
        "mean-mode DiTs are rejected");
  Check(!vc::ParseConfig(
            Replace(sample, "\"audio_vae_config\": {",
                    "\"audio_vae_config\": {\"use_noise_block\": true,")),
        "noise blocks are rejected");
  const auto other_heads = vc::ParseConfig(
      Replace(sample, "\"kv_channels\": 128", "\"kv_channels\": 64"));
  Check(other_heads.has_value() && !vc::IsSupported(*other_heads),
        "head sizes other than 128 are unsupported");
}

}  // namespace

int main() {
  TestReleasedConfig();
  TestDefaultsAndErrors();
  std::cout << (failures == 0 ? "ok\n" : "FAIL\n");
  return failures == 0 ? 0 : 1;
}
