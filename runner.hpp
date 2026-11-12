#ifndef GUFO_MODELS_VOXCPM_RUNNER_HPP_
#define GUFO_MODELS_VOXCPM_RUNNER_HPP_

#include <gufo/ext/tts.hpp>
#include <memory>
#include <string>

namespace gufo::models::voxcpm {

[[nodiscard]] std::unique_ptr<ext::TtsRunner> CreateVoxCpm2Runner(
    const ext::TtsRunnerOptions& options, std::string* error);

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_RUNNER_HPP_
