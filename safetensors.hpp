#ifndef GUFO_MODELS_VOXCPM_SAFETENSORS_HPP_
#define GUFO_MODELS_VOXCPM_SAFETENSORS_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gufo/ext/platform/mapped_file.hpp>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::voxcpm {

/// Element types. kQ8_0 (GGML: blocks of 32 int8 values with one float16
/// scale, 34 bytes) only comes from GGUF files and has no element size.
enum class DType : std::uint8_t { kF32, kBF16, kF16, kI64, kI32, kU8, kQ8_0 };

[[nodiscard]] std::size_t DTypeSize(DType dtype);
[[nodiscard]] std::string_view DTypeName(DType dtype);

/// One tensor of a mapped safetensors file. `data` stays valid while the
/// file is open; `offset` is its byte position in the file.
struct Tensor {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<std::size_t> shape;
  const std::byte* data = nullptr;
  std::size_t bytes = 0;
  std::uint64_t offset = 0;

  [[nodiscard]] std::size_t Elements() const;
};

/// A read-only mapping of a safetensors file. Every header entry must be
/// well formed and lie inside the payload; tensors may not overlap.
class SafetensorsFile {
public:
  [[nodiscard]] static std::unique_ptr<SafetensorsFile> Open(
      const std::filesystem::path& path, std::string* error = nullptr);

  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;
  ~SafetensorsFile();

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }
  [[nodiscard]] const std::vector<Tensor>& tensors() const { return tensors_; }
  [[nodiscard]] const Tensor* Find(std::string_view name) const;

  /// The tensor `name` with exactly this type and shape, or null with the
  /// mismatch in `error`.
  [[nodiscard]] const Tensor* Expect(std::string_view name, DType dtype,
                                     std::initializer_list<std::size_t> shape,
                                     std::string* error = nullptr) const;

  /// Drops resident pages once the weights live on the device; `data`
  /// pointers stay valid and fault the pages back in on access.
  void ReleasePages() const noexcept;

private:
  SafetensorsFile() = default;

  std::filesystem::path path_;
  platform::ReadOnlyMappedFile mapping_;
  std::vector<Tensor> tensors_;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_SAFETENSORS_HPP_
