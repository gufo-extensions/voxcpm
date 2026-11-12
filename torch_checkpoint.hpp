#ifndef GUFO_MODELS_VOXCPM_TORCH_CHECKPOINT_HPP_
#define GUFO_MODELS_VOXCPM_TORCH_CHECKPOINT_HPP_

#include <filesystem>
#include <gufo/ext/platform/mapped_file.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "safetensors.hpp"

namespace gufo::models::voxcpm {

/// A read-only mapping of a state dict saved by `torch.save` in the zip
/// format (PyTorch >= 1.6), such as VoxCPM2's `audiovae.pth`. The pickle is
/// interpreted, never executed: only the opcodes and globals a tensor state
/// dict needs are accepted (OrderedDict, `torch._utils._rebuild_tensor_v2`
/// and the typed storages), every tensor must be contiguous and lie inside
/// its stored, uncompressed storage. A top-level dict with a "state_dict"
/// entry is unwrapped, as VoxCPM's loader does.
class TorchCheckpoint {
public:
  [[nodiscard]] static std::unique_ptr<TorchCheckpoint> Open(
      const std::filesystem::path& path, std::string* error = nullptr);
  TorchCheckpoint(const TorchCheckpoint&) = delete;
  TorchCheckpoint& operator=(const TorchCheckpoint&) = delete;
  ~TorchCheckpoint();

  [[nodiscard]] const std::vector<Tensor>& tensors() const { return tensors_; }
  [[nodiscard]] const Tensor* Find(std::string_view name) const;

private:
  TorchCheckpoint() = default;

  platform::ReadOnlyMappedFile mapping_;
  std::vector<Tensor> tensors_;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_TORCH_CHECKPOINT_HPP_
