#include "model_files.hpp"

#include <cstdint>
#include <fstream>
#include <gufo/ext/gguf_reader.hpp>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <variant>

#include "torch_checkpoint.hpp"

namespace gufo::models::voxcpm {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kModelPrefix = "weights/";
constexpr std::string_view kVaePrefix = "audiovae_weights/";
constexpr std::string_view kAbbreviated = "_audiocpp.";

class FilesError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

std::string ReadFile(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw FilesError("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

std::optional<DType> FromGgml(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kF32:
      return DType::kF32;
    case core::GgmlType::kF16:
      return DType::kF16;
    case core::GgmlType::kBF16:
      return DType::kBF16;
    case core::GgmlType::kQ8_0:
      return DType::kQ8_0;
    case core::GgmlType::kI32:
      return DType::kI32;
    default:
      return std::nullopt;
  }
}

/// The integers of a metadata array, whatever their signedness.
std::vector<std::uint64_t> Integers(const core::GgufReader& reader,
                                    std::string_view key) {
  const core::GgufMetadataValue* value = reader.FindMetadata(key);
  if (value == nullptr)
    throw FilesError(std::string("GGUF lacks ") + std::string(key));
  if (const auto* unsigned_values =
          std::get_if<std::vector<std::uint64_t>>(&value->value))
    return *unsigned_values;
  if (const auto* signed_values =
          std::get_if<std::vector<std::int64_t>>(&value->value))
    return {signed_values->begin(), signed_values->end()};
  throw FilesError(std::string("GGUF ") + std::string(key) +
                   " is not an integer array");
}

/// An embedded file of an audio.cpp GGUF.
std::string Embedded(const core::GgufReader& reader, std::string_view name) {
  const std::vector<std::string_view> names =
      reader.GetMetadataStringArray("audiocpp.embedded_files.names");
  const std::vector<std::uint64_t> offsets =
      Integers(reader, "audiocpp.embedded_files.offsets");
  const std::vector<std::uint64_t> data =
      Integers(reader, "audiocpp.embedded_files.data");
  for (std::size_t index = 0; index < names.size(); ++index) {
    if (names[index] != name)
      continue;
    if (index + 1 >= offsets.size() || offsets[index] > offsets[index + 1] ||
        offsets[index + 1] > data.size())
      throw FilesError("GGUF embedded file table is malformed");
    std::string text;
    text.reserve(offsets[index + 1] - offsets[index]);
    for (std::uint64_t at = offsets[index]; at < offsets[index + 1]; ++at)
      text.push_back(static_cast<char>(data[at]));
    return text;
  }
  throw FilesError("GGUF lacks the embedded " + std::string(name));
}

bool IsVoxCpm2(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture") ==
             std::optional<std::string_view>("audiocpp") &&
         reader.GetMetadataString("audiocpp.model_spec.family") ==
             std::optional<std::string_view>("voxcpm2");
}

}  // namespace

ModelFiles::ModelFiles() = default;
ModelFiles::~ModelFiles() = default;

bool ModelFiles::IsVoxCpm2Gguf(const fs::path& path) {
  std::error_code code;
  if (!fs::is_regular_file(path, code) || path.extension() != ".gguf")
    return false;
  const auto reader = core::GgufReader::OpenFile(path);
  return reader != nullptr && IsVoxCpm2(*reader);
}

std::unique_ptr<ModelFiles> ModelFiles::Open(const fs::path& path,
                                             std::string* error) {
  std::unique_ptr<ModelFiles> files(new ModelFiles());
  try {
    std::error_code code;
    if (fs::is_directory(path, code)) {
      files->config_ = ReadFile(path / "config.json");
      files->tokenizer_ = ReadFile(path / "tokenizer.json");
      std::string open_error;
      files->safetensors_ =
          SafetensorsFile::Open(path / "model.safetensors", &open_error);
      if (files->safetensors_ == nullptr)
        throw FilesError(open_error);
      files->checkpoint_ =
          TorchCheckpoint::Open(path / "audiovae.pth", &open_error);
      if (files->checkpoint_ == nullptr)
        throw FilesError(open_error);
      files->model_ = files->safetensors_->tensors();
      files->vae_ = files->checkpoint_->tensors();
    } else {
      std::string open_error;
      files->gguf_ = core::GgufReader::OpenFile(path, &open_error);
      if (files->gguf_ == nullptr)
        throw FilesError(path.string() + ": " + open_error);
      const core::GgufReader& reader = *files->gguf_;
      if (!IsVoxCpm2(reader))
        throw FilesError(path.string() + " is not an audio.cpp VoxCPM2 GGUF");
      files->config_ = Embedded(reader, "config.json");
      files->tokenizer_ = Embedded(reader, "tokenizer.json");
      const std::vector<std::string_view> names =
          reader.GetMetadataStringArray("audiocpp.tensor_names");
      for (const core::GgufTensorInfo& info : reader.GetTensors()) {
        std::string name(info.name);
        if (name.starts_with(kAbbreviated)) {
          const std::size_t index =
              std::stoul(name.substr(kAbbreviated.size()));
          if (index >= names.size())
            throw FilesError("GGUF tensor " + name + " has no listed name");
          name = std::string(names[index]);
        }
        const auto dtype = FromGgml(info.type);
        if (!dtype.has_value())
          throw FilesError("GGUF tensor " + name + " has an unsupported type");
        Tensor tensor;
        tensor.dtype = *dtype;
        // GGML lists dimensions innermost first.
        for (auto it = info.dimensions.rbegin(); it != info.dimensions.rend();
             ++it)
          tensor.shape.push_back(static_cast<std::size_t>(*it));
        tensor.data = static_cast<const std::byte*>(info.data);
        tensor.bytes = info.size_bytes;
        tensor.offset = info.offset;
        if (name.starts_with(kModelPrefix)) {
          tensor.name = name.substr(kModelPrefix.size());
          files->model_.push_back(std::move(tensor));
        } else if (name.starts_with(kVaePrefix)) {
          tensor.name = name.substr(kVaePrefix.size());
          files->vae_.push_back(std::move(tensor));
        } else {
          throw FilesError("GGUF tensor " + name + " belongs to no component");
        }
      }
    }
    for (std::size_t index = 0; index < files->model_.size(); ++index)
      files->model_index_.emplace(files->model_[index].name, index);
    for (std::size_t index = 0; index < files->vae_.size(); ++index)
      files->vae_index_.emplace(files->vae_[index].name, index);
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return nullptr;
  }
  return files;
}

const Tensor* ModelFiles::Model(std::string_view name) const {
  const auto found = model_index_.find(std::string(name));
  return found == model_index_.end() ? nullptr : &model_[found->second];
}

const Tensor* ModelFiles::Vae(std::string_view name) const {
  const auto found = vae_index_.find(std::string(name));
  return found == vae_index_.end() ? nullptr : &vae_[found->second];
}

void ModelFiles::ReleasePages() const {
  if (safetensors_ != nullptr)
    safetensors_->ReleasePages();
}

}  // namespace gufo::models::voxcpm
