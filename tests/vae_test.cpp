// AudioVAE V2 on the GPU against the float32 reference: decoding the
// zero-shot latents in patches (streamed) and at once, and encoding the
// synthetic 16 kHz signal.
#include "vae.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "model_files.hpp"
#include "safetensors.hpp"

namespace fs = std::filesystem;
namespace vc = gufo::models::voxcpm;

namespace {

constexpr int kSkip = 77;
int failures = 0;
// The reference runs float32 weights; GGUFs may store some in float16 or
// bfloat16 (audio.cpp's bf16 file holds the whole AudioVAE in bfloat16).
double limit_scale = 1.0;

std::vector<float> Get(const vc::SafetensorsFile& file,
                       const std::string& name) {
  const vc::Tensor* tensor = file.Find(name);
  if (tensor == nullptr)
    throw std::runtime_error("golden lacks " + name);
  std::vector<float> values(tensor->Elements());
  std::memcpy(values.data(), tensor->data, tensor->bytes);
  return values;
}

std::vector<float> Transpose(const std::vector<float>& values, std::size_t rows,
                             std::size_t columns) {
  std::vector<float> result(values.size());
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < columns; ++c)
      result[c * rows + r] = values[r * columns + c];
  return result;
}

void Compare(const std::string& name, const std::vector<float>& actual,
             const std::vector<float>& expected, double limit) {
  if (actual.size() != expected.size()) {
    std::cout << "FAIL " << name << ": " << actual.size() << " values, "
              << expected.size() << " expected\n";
    ++failures;
    return;
  }
  double error = 0.0;
  double norm = 0.0;
  double worst = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double difference = actual[i] - expected[i];
    error += difference * difference;
    norm += static_cast<double>(expected[i]) * expected[i];
    worst = std::max(worst, std::fabs(difference));
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  limit *= limit_scale;
  const bool ok = std::isfinite(relative) && relative <= limit;
  std::cout << (ok ? "ok   " : "FAIL ") << std::left << std::setw(20) << name
            << " rel " << std::scientific << std::setprecision(2) << relative
            << " max " << worst << " (limit " << limit << ")\n"
            << std::defaultfloat;
  if (!ok)
    ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  // "gguf" reads the checkpoint from an audio.cpp GGUF instead of the
  // Hugging Face directory.
  const bool gguf = argc > 1 && std::string(argv[1]) == "gguf";
  const char* variable = gguf ? "GUFO_VOXCPM_GGUF" : "GUFO_VOXCPM_MODEL";
  const char* root = std::getenv(variable);
  if (root == nullptr || *root == '\0') {
    std::cout << "skip: " << variable << " is not set\n";
    return kSkip;
  }
  std::string error;
  const auto files = vc::ModelFiles::Open(root, &error);
  if (files == nullptr) {
    std::cout << "FAIL files: " << error << '\n';
    return 1;
  }
  const auto config = vc::ParseConfig(files->config(), &error);
  const vc::Tensor* weight = files->Vae("decoder.model.1.weight_v");
  const vc::Tensor* embed =
      files->Vae("decoder.sr_cond_model.2.scale_embed.weight");
  if (weight != nullptr && weight->dtype != vc::DType::kF32)
    limit_scale = 200.0;
  else if (embed != nullptr && embed->dtype != vc::DType::kF32)
    limit_scale = 10.0;
  const auto golden = vc::SafetensorsFile::Open(
      fs::path(GUFO_VOXCPM_FIXTURES) / "generator_golden.safetensors", &error);
  if (!config.has_value() || golden == nullptr) {
    std::cout << "FAIL setup: " << error << '\n';
    return 1;
  }
  const auto vae = vc::AudioVae::Create(*config, *files, &error);
  if (vae == nullptr) {
    std::cout << "FAIL AudioVAE: " << error << '\n';
    return 1;
  }
  try {
    const std::size_t latent = config->vae.latent_dim;
    const std::vector<float> latents =
        Transpose(Get(*golden, "zs.latents"), latent,
                  Get(*golden, "zs.latents").size() / latent);
    const std::size_t frames = latents.size() / latent;
    const std::vector<float> expected = Get(*golden, "vae.decoded");

    std::vector<float> streamed;
    vae->Reset();
    for (std::size_t frame = 0; frame < frames; frame += config->patch_size)
      vae->Decode(latents.data() + frame * latent, config->patch_size,
                  &streamed);
    Compare("decode streamed", streamed, expected, 1e-4);

    std::vector<float> whole;
    vae->Reset();
    vae->Decode(latents.data(), frames, &whole);
    Compare("decode at once", whole, expected, 1e-4);

    const std::vector<float> signal = Get(*golden, "vae.signal");
    const std::vector<float> encoded = vae->Encode(signal);
    const std::vector<float> reference = Get(*golden, "vae.encoded");
    Compare("encode", encoded,
            Transpose(reference, latent, reference.size() / latent), 1e-4);
  } catch (const std::exception& failure) {
    std::cout << "FAIL " << failure.what() << '\n';
    return 1;
  }
  std::cout << (failures == 0 ? "ok\n" : "FAIL\n");
  return failures == 0 ? 0 : 1;
}
