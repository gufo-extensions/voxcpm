#include <gufo/ext/generated/voxcpm_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include "probe.hpp"
#include "runner.hpp"

void voxcpm2_provider_register(gufo::ext::Registry& registry) {
  namespace voxcpm = gufo::models::voxcpm;
  registry.AddProvider(
      gufo::ext::generated::extension_voxcpm::kProvider_voxcpm2);
  registry.AddTtsProvider({
      .id = "voxcpm2",
      .probe = &voxcpm::ProbeVoxCpm2,
      .resolve = &voxcpm::ResolveVoxCpm2Artifacts,
      .create = nullptr,
      .create_with_options = &voxcpm::CreateVoxCpm2Runner,
  });
}
