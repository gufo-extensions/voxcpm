// The GPU generator against the float32 reference run of
// tools/reference/generator_golden.py: prefill, three zero-shot patches
// with the reference's noise, a reference-mode prompt and one DiT call.
#include "generator.hpp"

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
// Q8_0 weights (audio.cpp's quantized GGUF) move every tensor by about 1 %
// against the bfloat16 reference; their limits are five times wider.
double tolerance = 1.0;

class Golden {
public:
  explicit Golden(const vc::SafetensorsFile& file) : file_(file) {}

  [[nodiscard]] bool Has(const std::string& name) const {
    return file_.Find(name) != nullptr;
  }

  [[nodiscard]] std::vector<float> Get(const std::string& name) const {
    const vc::Tensor* tensor = file_.Find(name);
    if (tensor == nullptr || tensor->dtype != vc::DType::kF32)
      throw std::runtime_error("golden lacks " + name);
    std::vector<float> values(tensor->Elements());
    std::memcpy(values.data(), tensor->data, tensor->bytes);
    return values;
  }

  [[nodiscard]] std::vector<std::int32_t> Ints(const std::string& name) const {
    std::vector<std::int32_t> values;
    for (const float value : Get(name))
      values.push_back(static_cast<std::int32_t>(std::lround(value)));
    return values;
  }

private:
  const vc::SafetensorsFile& file_;
};

/// [rows][columns] -> [columns][rows] for each of `batch` matrices.
std::vector<float> Transpose(const std::vector<float>& values, std::size_t rows,
                             std::size_t columns, std::size_t batch = 1) {
  std::vector<float> result(values.size());
  for (std::size_t b = 0; b < batch; ++b)
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < columns; ++c)
        result[b * rows * columns + c * rows + r] =
            values[b * rows * columns + r * columns + c];
  return result;
}

/// Relative L2 error of `actual` against `expected`; fails above `limit`.
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
  limit *= tolerance;
  const bool ok = std::isfinite(relative) && relative <= limit;
  std::cout << (ok ? "ok   " : "FAIL ") << std::left << std::setw(26) << name
            << " rel " << std::scientific << std::setprecision(2) << relative
            << " max " << worst << " (limit " << limit << ")\n"
            << std::defaultfloat;
  if (!ok)
    ++failures;
}

vc::Prompt PromptOf(const Golden& golden, const std::string& prefix) {
  vc::Prompt prompt;
  prompt.tokens = golden.Ints(prefix + ".text_token");
  prompt.text_mask = golden.Ints(prefix + ".text_mask");
  prompt.audio_mask = golden.Ints(prefix + ".feat_mask");
  prompt.feats = golden.Get(prefix + ".feat_in");
  return prompt;
}

void RunCase(vc::Generator& generator, const Golden& golden,
             const std::string& prefix, std::size_t patches) {
  const vc::Config& config = generator.config();
  const std::size_t patch = config.patch_size;
  const std::size_t feat = config.feat_dim;
  vc::Trace trace;
  generator.set_trace(&trace);
  generator.Prefill(PromptOf(golden, prefix));
  Compare(prefix + " lm prefill", trace.lm_prefill,
          golden.Get(prefix + ".lm.0"), 1e-2);
  Compare(prefix + " residual prefill", trace.residual_prefill,
          golden.Get(prefix + ".residual.0"), 1e-2);
  for (std::size_t step = 0; step < patches; ++step) {
    const std::string k = std::to_string(step);
    // The first patch starts from the reference's prompt; later ones carry
    // the earlier patches' deviations (bf16 GEMM operands against the
    // float32 reference) through the autoregressive state.
    const double limit = step == 0 ? 1e-2 : 1e-1;
    const std::vector<float> noise =
        Transpose(golden.Get(prefix + ".noise." + k), feat, patch);
    std::vector<float> latent(patch * feat);
    std::vector<float> stop(2);
    generator.Step(noise.data(), 10, 2.0F, latent.data(), stop.data());
    Compare(prefix + " mu " + k, trace.mu.back(),
            golden.Get(prefix + ".mu." + k), limit);
    Compare(prefix + " stop " + k, stop, golden.Get(prefix + ".stop." + k),
            limit);
    Compare(prefix + " latent " + k, latent,
            Transpose(golden.Get(prefix + ".feat." + k), feat, patch),
            2 * limit);
    // The LM steps take this patch's latent, so they already carry its
    // deviation.
    Compare(prefix + " lm step " + k, trace.lm_step.back(),
            golden.Get(prefix + ".lm_step." + k), 1e-1);
    Compare(prefix + " residual step " + k, trace.residual_step.back(),
            golden.Get(prefix + ".residual_step." + k), 1e-1);
  }
  generator.set_trace(nullptr);
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
  const vc::Tensor* sample =
      files->Model("base_lm.layers.0.mlp.down_proj.weight");
  if (sample != nullptr && sample->dtype == vc::DType::kQ8_0)
    tolerance = 5.0;
  if (!config.has_value()) {
    std::cout << "FAIL config: " << error << '\n';
    return 1;
  }
  const auto file = vc::SafetensorsFile::Open(
      fs::path(GUFO_VOXCPM_FIXTURES) / "generator_golden.safetensors", &error);
  if (file == nullptr) {
    std::cout << "FAIL golden: " << error << '\n';
    return 1;
  }
  const Golden golden(*file);
  const auto generator = vc::Generator::Create(*config, *files, &error);
  if (generator == nullptr) {
    std::cout << "FAIL generator: " << error << '\n';
    return 1;
  }
  try {
    const std::size_t patch = config->patch_size;
    const std::size_t feat = config->feat_dim;
    // One DiT evaluation with its CFG pair.
    const std::vector<float> mu = golden.Get("dit.mu");
    const std::vector<float> x = Transpose(golden.Get("dit.x"), feat, patch);
    const std::vector<float> cond =
        Transpose(golden.Get("dit.cond"), feat, patch);
    std::vector<float> velocity(2 * patch * feat);
    generator->Velocity(mu.data(), cond.data(), x.data(),
                        golden.Get("dit.t")[0], velocity.data());
    Compare("dit velocity", velocity,
            Transpose(golden.Get("dit.out"), feat, patch, 2), 1e-2);

    RunCase(*generator, golden, "zs", 3);
    RunCase(*generator, golden, "ref", 1);
  } catch (const std::exception& failure) {
    std::cout << "FAIL " << failure.what() << '\n';
    return 1;
  }
  std::cout << (failures == 0 ? "ok\n" : "FAIL\n");
  return failures == 0 ? 0 : 1;
}
