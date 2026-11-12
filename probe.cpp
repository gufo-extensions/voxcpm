#include "probe.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <system_error>

#include "config.hpp"
#include "model_files.hpp"

namespace gufo::models::voxcpm {

bool ProbeVoxCpm2(std::string_view model_root) {
  std::error_code code;
  const std::filesystem::path root(model_root);
  if (ModelFiles::IsVoxCpm2Gguf(root)) {
    const auto files = ModelFiles::Open(root);
    if (files == nullptr)
      return false;
    const auto config = ParseConfig(files->config());
    return config.has_value() && IsSupported(*config);
  }
  if (!std::filesystem::is_directory(root, code) ||
      !std::filesystem::is_regular_file(root / "config.json", code))
    return false;
  const auto config = LoadConfig(model_root);
  return config.has_value() && IsSupported(*config);
}

bool ResolveVoxCpm2Artifacts(std::string_view model_root, std::string* error) {
  const auto fail = [error](const std::string& message) {
    if (error != nullptr)
      *error = message;
    return false;
  };
  const std::filesystem::path root(model_root);
  if (ModelFiles::IsVoxCpm2Gguf(root)) {
    const auto files = ModelFiles::Open(root, error);
    if (files == nullptr)
      return false;
    const auto config = ParseConfig(files->config(), error);
    if (!config.has_value() || !IsSupported(*config, error))
      return false;
    if (files->Model("base_lm.embed_tokens.weight") == nullptr ||
        files->Vae("decoder.model.0.weight_v") == nullptr)
      return fail("the GGUF lacks VoxCPM2's generator or AudioVAE tensors");
    return true;
  }
  const auto config = LoadConfig(model_root, error);
  if (!config.has_value() || !IsSupported(*config, error))
    return false;
  const auto missing = [&root](const char* file) {
    std::error_code code;
    return !std::filesystem::is_regular_file(root / file, code);
  };
  constexpr std::array<const char*, 3> kFiles = {
      "model.safetensors", "tokenizer.json", "audiovae.pth"};
  for (const char* file : kFiles)
    if (missing(file))
      return fail("VoxCPM2 model directory lacks " + std::string(file));
  return true;
}

}  // namespace gufo::models::voxcpm
