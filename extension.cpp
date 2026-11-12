#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include "probe.hpp"

// Without HIP the provider's library isn't built; registering its probe
// still lets Gufo name the model it can't run here.
void voxcpm_register(gufo::ext::Registry& registry) {
#if !defined(ENGINE_ENABLE_HIP)
  registry.AddTtsProvider({
      .id = "voxcpm2",
      .probe = &gufo::models::voxcpm::ProbeVoxCpm2,
      .resolve = &gufo::models::voxcpm::ResolveVoxCpm2Artifacts,
      .create = nullptr,
  });
#else
  (void)registry;
#endif
}
