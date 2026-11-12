#ifndef GUFO_MODELS_VOXCPM_PROBE_HPP_
#define GUFO_MODELS_VOXCPM_PROBE_HPP_

#include <string>
#include <string_view>

namespace gufo::models::voxcpm {

/// `model_root` is a directory holding a supported VoxCPM2 checkpoint in the
/// openbmb Hugging Face layout (`config.json` with architecture "voxcpm2"),
/// or an audio.cpp VoxCPM2 GGUF.
[[nodiscard]] bool ProbeVoxCpm2(std::string_view model_root);
/// Checks that the configuration is supported and the weight and tokenizer
/// files exist.
[[nodiscard]] bool ResolveVoxCpm2Artifacts(std::string_view model_root,
                                           std::string* error = nullptr);

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_PROBE_HPP_
