#include "probe.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "config_samples.hpp"

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

void Write(const fs::path& path, std::string_view text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

/// A fresh directory with a VoxCPM2 configuration.
class SampleDirectory {
public:
  explicit SampleDirectory(std::string_view config) {
    root_ = fs::temp_directory_path() /
            ("voxcpm_probe_test_" + std::to_string(::getpid()));
    fs::remove_all(root_);
    Write(root_ / "config.json", config);
  }
  ~SampleDirectory() {
    std::error_code ignored;
    fs::remove_all(root_, ignored);
  }
  SampleDirectory(const SampleDirectory&) = delete;
  SampleDirectory& operator=(const SampleDirectory&) = delete;

  [[nodiscard]] std::string root() const { return root_.string(); }
  void Add(const char* file) const { Write(root_ / file, ""); }

private:
  fs::path root_;
};

int TestSamples() {
  {
    const SampleDirectory directory(voxcpm_test::kModelConfig);
    std::string error;
    Check(vc::ProbeVoxCpm2(directory.root()), "probe accepts VoxCPM2");
    Check(!vc::ResolveVoxCpm2Artifacts(directory.root(), &error) &&
              error.find("model.safetensors") != std::string::npos,
          "resolve names the missing weights");
    directory.Add("model.safetensors");
    directory.Add("tokenizer.json");
    Check(!vc::ResolveVoxCpm2Artifacts(directory.root(), &error) &&
              error.find("audiovae.pth") != std::string::npos,
          "resolve names the missing AudioVAE");
    directory.Add("audiovae.pth");
    Check(vc::ResolveVoxCpm2Artifacts(directory.root(), &error),
          "resolve accepts the complete directory: " + error);
  }
  {
    std::string config = voxcpm_test::kModelConfig;
    config.replace(config.find("\"voxcpm2\""), 9, "\"voxcpm\"");
    const SampleDirectory directory(config);
    Check(!vc::ProbeVoxCpm2(directory.root()), "probe rejects VoxCPM 1");
  }
  Check(!vc::ProbeVoxCpm2("/nonexistent/voxcpm"),
        "probe rejects a missing directory");
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
  Check(vc::ProbeVoxCpm2(root), "probe accepts the model");
  Check(vc::ResolveVoxCpm2Artifacts(root, &error),
        "resolve accepts the model: " + error);
  // An audio.cpp GGUF, when one is given, is accepted as well.
  const char* gguf = std::getenv("GUFO_VOXCPM_GGUF");
  if (gguf != nullptr && *gguf != '\0') {
    Check(vc::ProbeVoxCpm2(gguf), "probe accepts the GGUF");
    Check(vc::ResolveVoxCpm2Artifacts(gguf, &error),
          "resolve accepts the GGUF: " + error);
  }
  std::cout << (failures == 0 ? "ok\n" : "FAIL\n");
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "samples";
  return mode == "model" ? TestModel() : TestSamples();
}
