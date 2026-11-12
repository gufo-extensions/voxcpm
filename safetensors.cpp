// Derived from Gufo's Qwen3-TTS safetensors loader (MIT, see
// licenses/gufo.txt); this version rejects malformed entries instead of
// skipping them.
#include "safetensors.hpp"

#include <algorithm>
#include <cstring>
#include <gufo/ext/json.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace gufo::models::voxcpm {
namespace {

namespace json = gufo::json;

class FormatError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

std::optional<DType> ParseDType(std::string_view name) {
  if (name == "F32")
    return DType::kF32;
  if (name == "BF16")
    return DType::kBF16;
  if (name == "F16")
    return DType::kF16;
  if (name == "I64")
    return DType::kI64;
  if (name == "I32")
    return DType::kI32;
  if (name == "U8")
    return DType::kU8;
  return std::nullopt;
}

std::uint64_t ReadLittleEndian64(const std::uint8_t* bytes) {
  std::uint64_t value = 0;
  for (int index = 7; index >= 0; --index)
    value = (value << 8U) | bytes[index];
  return value;
}

std::string ShapeText(const std::vector<std::size_t>& shape) {
  std::string text = "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0)
      text += ", ";
    text += std::to_string(shape[index]);
  }
  return text + "]";
}

Tensor ParseEntry(const std::string& name, const json::Value& entry,
                  const std::byte* payload, std::uint64_t payload_offset,
                  std::uint64_t payload_bytes) {
  if (!entry.is_object())
    throw FormatError(name + ": header entry is not an object");
  const auto dtype = ParseDType(entry.member_str("dtype"));
  if (!dtype.has_value())
    throw FormatError(name + ": unsupported dtype \"" +
                      entry.member_str("dtype") + "\"");
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = *dtype;
  const json::Value* shape = entry.find("shape");
  if (shape == nullptr || !shape->is_array())
    throw FormatError(name + ": missing shape");
  std::size_t elements = 1;
  for (const json::Value& dimension : shape->items()) {
    constexpr std::size_t kInvalid = std::numeric_limits<std::size_t>::max();
    const std::size_t size = dimension.as_size(kInvalid);
    if (size == kInvalid)
      throw FormatError(name + ": malformed shape");
    if (size != 0 && elements > std::numeric_limits<std::size_t>::max() / size)
      throw FormatError(name + ": shape overflows");
    elements *= size;
    tensor.shape.push_back(size);
  }
  const json::Value* offsets = entry.find("data_offsets");
  if (offsets == nullptr || !offsets->is_array() || offsets->size() != 2)
    throw FormatError(name + ": missing data_offsets");
  constexpr std::uint64_t kInvalid = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t begin = offsets->items()[0].as_size(kInvalid);
  const std::uint64_t end = offsets->items()[1].as_size(kInvalid);
  if (begin == kInvalid || end == kInvalid || begin > end ||
      end > payload_bytes)
    throw FormatError(name + ": data_offsets lie outside the payload");
  const std::size_t element_bytes = DTypeSize(tensor.dtype);
  if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes ||
      elements * element_bytes != end - begin)
    throw FormatError(name + ": " + std::to_string(end - begin) +
                      " bytes don't hold shape " + ShapeText(tensor.shape));
  tensor.data = payload + begin;
  tensor.bytes = static_cast<std::size_t>(end - begin);
  tensor.offset = payload_offset + begin;
  return tensor;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

}  // namespace

std::size_t DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kF32:
    case DType::kI32:
      return 4;
    case DType::kBF16:
    case DType::kF16:
      return 2;
    case DType::kI64:
      return 8;
    case DType::kU8:
      return 1;
    case DType::kQ8_0:
      return 0;
  }
  return 0;
}

std::string_view DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kF32:
      return "F32";
    case DType::kBF16:
      return "BF16";
    case DType::kF16:
      return "F16";
    case DType::kI64:
      return "I64";
    case DType::kI32:
      return "I32";
    case DType::kU8:
      return "U8";
    case DType::kQ8_0:
      return "Q8_0";
  }
  return "?";
}

std::size_t Tensor::Elements() const {
  std::size_t elements = 1;
  for (const std::size_t size : shape)
    elements *= size;
  return elements;
}

SafetensorsFile::~SafetensorsFile() {
  mapping_.Close();
}

std::unique_ptr<SafetensorsFile> SafetensorsFile::Open(
    const std::filesystem::path& path, std::string* error) {
  std::unique_ptr<SafetensorsFile> file(new SafetensorsFile());
  file->path_ = path;
  std::string open_error;
  file->mapping_ = platform::ReadOnlyMappedFile::Open(path, &open_error);
  if (!file->mapping_.valid()) {
    SetError(error, "cannot map " + path.string() + ": " + open_error);
    return nullptr;
  }
  try {
    const std::uint8_t* bytes = file->mapping_.data();
    const std::size_t size = file->mapping_.size();
    if (size < 8)
      throw FormatError("file is shorter than its header length");
    const std::uint64_t header_bytes = ReadLittleEndian64(bytes);
    // The format caps headers at 100 MB.
    if (header_bytes == 0 || header_bytes > 100'000'000 ||
        header_bytes > size - 8)
      throw FormatError("implausible header length");
    json::Value header;
    try {
      header = json::parse(std::string_view(
          reinterpret_cast<const char*>(bytes + 8), header_bytes));
    } catch (const std::exception& parse_error) {
      throw FormatError(std::string("header: ") + parse_error.what());
    }
    if (!header.is_object())
      throw FormatError("header is not a JSON object");
    const std::uint64_t payload_offset = 8 + header_bytes;
    const auto* payload =
        reinterpret_cast<const std::byte*>(bytes + payload_offset);
    for (const auto& [name, entry] : header.members()) {
      if (name == "__metadata__")
        continue;
      file->tensors_.push_back(ParseEntry(name, entry, payload, payload_offset,
                                          size - payload_offset));
    }
    if (file->tensors_.empty())
      throw FormatError("no tensors");
    std::vector<const Tensor*> by_offset;
    for (const Tensor& tensor : file->tensors_)
      by_offset.push_back(&tensor);
    std::sort(by_offset.begin(), by_offset.end(),
              [](const Tensor* left, const Tensor* right) {
                return left->offset < right->offset;
              });
    for (std::size_t index = 1; index < by_offset.size(); ++index) {
      const Tensor& previous = *by_offset[index - 1];
      if (previous.offset + previous.bytes > by_offset[index]->offset)
        throw FormatError(previous.name + " overlaps " +
                          by_offset[index]->name);
    }
    std::sort(file->tensors_.begin(), file->tensors_.end(),
              [](const Tensor& left, const Tensor& right) {
                return left.name < right.name;
              });
  } catch (const FormatError& failure) {
    SetError(error, path.string() + ": " + failure.what());
    return nullptr;
  }
  return file;
}

const Tensor* SafetensorsFile::Find(std::string_view name) const {
  const auto found =
      std::lower_bound(tensors_.begin(), tensors_.end(), name,
                       [](const Tensor& tensor, std::string_view key) {
                         return tensor.name < key;
                       });
  return found != tensors_.end() && found->name == name ? &*found : nullptr;
}

const Tensor* SafetensorsFile::Expect(std::string_view name, DType dtype,
                                      std::initializer_list<std::size_t> shape,
                                      std::string* error) const {
  const Tensor* tensor = Find(name);
  if (tensor == nullptr) {
    SetError(error, path_.filename().string() + " lacks " + std::string(name));
    return nullptr;
  }
  const std::vector<std::size_t> expected(shape);
  if (tensor->dtype != dtype || tensor->shape != expected) {
    SetError(error, std::string(name) + " is " +
                        std::string(DTypeName(tensor->dtype)) + " " +
                        ShapeText(tensor->shape) + ", expected " +
                        std::string(DTypeName(dtype)) + " " +
                        ShapeText(expected));
    return nullptr;
  }
  return tensor;
}

void SafetensorsFile::ReleasePages() const noexcept {
  platform::ReleaseMappedPages(mapping_.data(), mapping_.size());
}

}  // namespace gufo::models::voxcpm
