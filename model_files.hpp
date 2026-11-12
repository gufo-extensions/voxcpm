#ifndef GUFO_MODELS_VOXCPM_MODEL_FILES_HPP_
#define GUFO_MODELS_VOXCPM_MODEL_FILES_HPP_

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "safetensors.hpp"
#include "torch_checkpoint.hpp"

namespace gufo::core {
class GgufReader;
}  // namespace gufo::core

namespace gufo::models::voxcpm {

/// Whether two shapes agree up to dimensions of size 1 (GGUF drops trailing
/// ones of GGML's order, so [1, C, 1] arrives as [C, 1]); the data layout is
/// the same.
[[nodiscard]] inline bool SameShape(const std::vector<std::size_t>& a,
                                    const std::vector<std::size_t>& b) {
  std::vector<std::size_t> x;
  std::vector<std::size_t> y;
  for (const std::size_t size : a)
    if (size != 1)
      x.push_back(size);
  for (const std::size_t size : b)
    if (size != 1)
      y.push_back(size);
  return x == y;
}

/// The files of a VoxCPM2 checkpoint, in either layout:
///  - the openbmb Hugging Face directory: config.json, tokenizer.json,
///    model.safetensors (bfloat16) and audiovae.pth;
///  - one audio.cpp GGUF (general.architecture "audiocpp", family
///    "voxcpm2"), which embeds those files' contents: the generator's
///    tensors under "weights/", AudioVAE's under "audiovae_weights/", names
///    too long for GGUF listed in audiocpp.tensor_names, linear weights in
///    BF16, F16, F32 or Q8_0.
/// Tensors keep the reference's names and PyTorch's dimension order.
class ModelFiles {
public:
  [[nodiscard]] static std::unique_ptr<ModelFiles> Open(
      const std::filesystem::path& path, std::string* error = nullptr);
  /// Whether `path` is a GGUF file of audio.cpp's VoxCPM2 layout (reads the
  /// header only).
  [[nodiscard]] static bool IsVoxCpm2Gguf(const std::filesystem::path& path);
  ~ModelFiles();
  ModelFiles(const ModelFiles&) = delete;
  ModelFiles& operator=(const ModelFiles&) = delete;

  [[nodiscard]] const std::string& config() const { return config_; }
  [[nodiscard]] const std::string& tokenizer() const { return tokenizer_; }
  [[nodiscard]] bool gguf() const { return gguf_ != nullptr; }

  /// A generator tensor (model.safetensors names) or an AudioVAE tensor
  /// (audiovae.pth names); null when absent.
  [[nodiscard]] const Tensor* Model(std::string_view name) const;
  [[nodiscard]] const Tensor* Vae(std::string_view name) const;
  [[nodiscard]] const std::vector<Tensor>& model_tensors() const {
    return model_;
  }

  /// Drops resident pages of the weight files once they live on the device.
  void ReleasePages() const;

private:
  ModelFiles();

  std::string config_;
  std::string tokenizer_;
  std::unique_ptr<SafetensorsFile> safetensors_;
  std::unique_ptr<TorchCheckpoint> checkpoint_;
  std::unique_ptr<core::GgufReader> gguf_;
  std::vector<Tensor> model_;
  std::vector<Tensor> vae_;
  std::unordered_map<std::string, std::size_t> model_index_;
  std::unordered_map<std::string, std::size_t> vae_index_;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_MODEL_FILES_HPP_
