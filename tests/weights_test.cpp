// The torch checkpoint reader on small torch.save files, and both weight
// files of the released model.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "safetensors.hpp"
#include "torch_checkpoint.hpp"

namespace fs = std::filesystem;
namespace vc = gufo::models::voxcpm;

namespace {

constexpr int kSkip = 77;
int failures = 0;

void Check(bool condition, const std::string& what) {
  if (!condition) {
    std::cout << "FAIL " << what << '\n';
    ++failures;
  }
}

template<typename T>
std::vector<T> Values(const vc::Tensor& tensor) {
  std::vector<T> values(tensor.bytes / sizeof(T));
  std::memcpy(values.data(), tensor.data, tensor.bytes);
  return values;
}

float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

int TestSamples(const fs::path& fixtures) {
  std::string error;
  const auto sample =
      vc::TorchCheckpoint::Open(fixtures / "sample.pth", &error);
  Check(sample != nullptr, "sample.pth opens: " + error);
  if (sample != nullptr) {
    Check(sample->tensors().size() == 6, "six tensors");
    const vc::Tensor* weight = sample->Find("layer.weight");
    Check(weight != nullptr && weight->dtype == vc::DType::kF32 &&
              weight->shape == std::vector<std::size_t>{2, 3} &&
              Values<float>(*weight) ==
                  std::vector<float>{1.5F, -2.0F, 3.25F, 0.0F, 4.0F, -0.5F},
          "float32 matrix");
    const vc::Tensor* bias = sample->Find("layer.bias");
    Check(bias != nullptr && bias->dtype == vc::DType::kI32 &&
              Values<std::int32_t>(*bias) ==
                  std::vector<std::int32_t>{7, -8, 9, 2147483647},
          "int32 vector");
    const vc::Tensor* half = sample->Find("half");
    Check(half != nullptr && half->dtype == vc::DType::kBF16 &&
              Bf16(Values<std::uint16_t>(*half)[0]) == 1.0F &&
              Bf16(Values<std::uint16_t>(*half)[1]) == -3.0F,
          "bfloat16 vector");
    const vc::Tensor* tail = sample->Find("tail");
    const vc::Tensor* head = sample->Find("head");
    Check(tail != nullptr && head != nullptr &&
              Values<float>(*tail) ==
                  std::vector<float>{6.0F, 7.0F, 8.0F, 9.0F} &&
              Values<float>(*head) == std::vector<float>{0.0F, 1.0F},
          "views at offsets into a shared storage");
    const vc::Tensor* scalar = sample->Find("scalar");
    Check(scalar != nullptr && scalar->shape.empty() &&
              Values<float>(*scalar) == std::vector<float>{0.25F},
          "zero-dimensional tensor");
  }
  Check(vc::TorchCheckpoint::Open(fixtures / "view.pth", &error) == nullptr &&
            error.find("not contiguous") != std::string::npos,
        "a transposed view is rejected");
  Check(vc::TorchCheckpoint::Open(fixtures / "tokenizer_golden.json", &error) ==
            nullptr,
        "a non-zip file is rejected");
  std::cout << (failures == 0 ? "ok\n" : "FAIL\n");
  return failures == 0 ? 0 : 1;
}

int TestModel() {
  const char* root = std::getenv("GUFO_VOXCPM_MODEL");
  if (root == nullptr || *root == '\0') {
    std::cout << "skip: GUFO_VOXCPM_MODEL is not set\n";
    return kSkip;
  }
  std::string error;
  const auto config = vc::LoadConfig(root, &error);
  Check(config.has_value(), "config loads: " + error);
  const auto model =
      vc::SafetensorsFile::Open(fs::path(root) / "model.safetensors", &error);
  Check(model != nullptr, "model.safetensors opens: " + error);
  if (model != nullptr && config.has_value()) {
    Check(model->tensors().size() == 577, "577 model tensors");
    const std::size_t hidden = config->lm.hidden;
    Check(model->Expect("base_lm.embed_tokens.weight", vc::DType::kBF16,
                        {config->lm.vocab, hidden}, &error) != nullptr,
          error);
    Check(model->Expect("feat_encoder.special_token", vc::DType::kBF16,
                        {1, 1, 1, config->encoder.hidden}, &error) != nullptr,
          error);
    Check(model->Expect("fusion_concat_proj.weight", vc::DType::kBF16,
                        {hidden, 2 * hidden}, &error) != nullptr,
          error);
  }
  const auto vae =
      vc::TorchCheckpoint::Open(fs::path(root) / "audiovae.pth", &error);
  Check(vae != nullptr, "audiovae.pth opens: " + error);
  if (vae != nullptr) {
    Check(vae->tensors().size() == 312, "312 AudioVAE tensors");
    const vc::Tensor* boundaries = vae->Find("decoder.sr_bin_boundaries");
    Check(boundaries != nullptr && boundaries->dtype == vc::DType::kI32 &&
              Values<std::int32_t>(*boundaries) ==
                  std::vector<std::int32_t>{20000, 30000, 40000},
          "rate bucket boundaries");
    const vc::Tensor* gain = vae->Find("encoder.fc_mu.weight_g");
    Check(gain != nullptr && gain->dtype == vc::DType::kF32 &&
              gain->shape == std::vector<std::size_t>{64, 1, 1},
          "weight-norm gains keep their shape");
    bool finite = true;
    for (const vc::Tensor& tensor : vae->tensors()) {
      if (tensor.dtype != vc::DType::kF32)
        continue;
      for (const float value : Values<float>(tensor))
        finite = finite && std::isfinite(value);
    }
    Check(finite, "AudioVAE weights are finite");
  }
  std::cout << (failures == 0 ? "ok\n" : "FAIL\n");
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "samples";
  return mode == "model" ? TestModel() : TestSamples(GUFO_VOXCPM_FIXTURES);
}
