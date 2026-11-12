#include "torch_checkpoint.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace gufo::models::voxcpm {
namespace {

class FormatError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw FormatError(message);
}

// ---------------------------------------------------------------- zip ---

struct Bytes {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;

  [[nodiscard]] std::uint64_t Read(std::size_t offset, int width) const {
    Require(offset + width <= size, "zip structure runs past the file");
    std::uint64_t value = 0;
    for (int index = width - 1; index >= 0; --index)
      value = (value << 8U) | data[offset + index];
    return value;
  }
};

struct Entry {
  std::uint64_t offset = 0;  // of the data in the file
  std::uint64_t size = 0;
};

/// Stored (uncompressed) entries of the zip archive by name.
std::map<std::string, Entry> ReadZip(const Bytes& file) {
  constexpr std::size_t kEndSize = 22;
  Require(file.size >= kEndSize, "not a zip archive");
  std::size_t end = file.size - kEndSize;
  const std::size_t lowest =
      file.size > kEndSize + 0xFFFF ? file.size - kEndSize - 0xFFFF : 0;
  while (file.Read(end, 4) != 0x06054b50U) {
    Require(end > lowest, "not a zip archive");
    --end;
  }
  std::uint64_t count = file.Read(end + 10, 2);
  std::uint64_t directory = file.Read(end + 16, 4);
  if (count == 0xFFFF || directory == 0xFFFFFFFFU) {
    Require(end >= 20 && file.Read(end - 20, 4) == 0x07064b50U,
            "zip64 locator missing");
    const std::uint64_t record = file.Read(end - 20 + 8, 8);
    Require(file.Read(record, 4) == 0x06064b50U, "zip64 record missing");
    count = file.Read(record + 32, 8);
    directory = file.Read(record + 48, 8);
  }
  std::map<std::string, Entry> entries;
  std::uint64_t at = directory;
  for (std::uint64_t index = 0; index < count; ++index) {
    Require(file.Read(at, 4) == 0x02014b50U, "bad zip directory entry");
    const auto method = file.Read(at + 10, 2);
    std::uint64_t compressed = file.Read(at + 20, 4);
    std::uint64_t size = file.Read(at + 24, 4);
    const auto name_size = file.Read(at + 28, 2);
    const auto extra_size = file.Read(at + 30, 2);
    const auto comment_size = file.Read(at + 32, 2);
    std::uint64_t local = file.Read(at + 42, 4);
    Require(at + 46 + name_size <= file.size, "zip name runs past the file");
    std::string name(reinterpret_cast<const char*>(file.data + at + 46),
                     name_size);
    // Zip64 extra field: the sizes and offset that overflowed, in order.
    for (std::uint64_t extra = at + 46 + name_size;
         extra + 4 <= at + 46 + name_size + extra_size;) {
      const auto id = file.Read(extra, 2);
      const auto length = file.Read(extra + 2, 2);
      if (id == 0x0001) {
        std::uint64_t field = extra + 4;
        if (size == 0xFFFFFFFFU) {
          size = file.Read(field, 8);
          field += 8;
        }
        if (compressed == 0xFFFFFFFFU) {
          compressed = file.Read(field, 8);
          field += 8;
        }
        if (local == 0xFFFFFFFFU)
          local = file.Read(field, 8);
      }
      extra += 4 + length;
    }
    Require(method == 0 && compressed == size,
            "zip entry " + name + " is compressed");
    Require(file.Read(local, 4) == 0x04034b50U, "bad zip local header");
    const std::uint64_t data =
        local + 30 + file.Read(local + 26, 2) + file.Read(local + 28, 2);
    Require(data <= file.size && size <= file.size - data,
            "zip entry " + name + " runs past the file");
    entries[name] = {data, size};
    at += 46 + name_size + extra_size + comment_size;
  }
  return entries;
}

// ------------------------------------------------------------- pickle ---

struct Object;
using Ref = std::shared_ptr<Object>;

struct Object {
  enum class Kind {
    kNone,
    kBool,
    kInt,
    kFloat,
    kString,
    kTuple,
    kList,
    kDict,
    kGlobal,
    kStorage,
    kTensor,
  };
  Kind kind = Kind::kNone;
  std::int64_t integer = 0;
  double number = 0.0;
  std::string text;  // string, "module name" of a global, storage key
  std::vector<Ref> items;
  std::vector<std::pair<Ref, Ref>> members;
  // Storage: element type and count. Tensor: storage, offset, size, stride.
  DType dtype = DType::kF32;
  std::uint64_t count = 0;
  Ref storage;
  std::vector<std::int64_t> size;
  std::vector<std::int64_t> stride;
};

Ref Make(Object::Kind kind) {
  auto object = std::make_shared<Object>();
  object->kind = kind;
  return object;
}

Ref MakeInt(std::int64_t value) {
  Ref object = Make(Object::Kind::kInt);
  object->integer = value;
  return object;
}

std::optional<DType> StorageType(const std::string& global) {
  if (global == "torch FloatStorage")
    return DType::kF32;
  if (global == "torch BFloat16Storage")
    return DType::kBF16;
  if (global == "torch HalfStorage")
    return DType::kF16;
  if (global == "torch LongStorage")
    return DType::kI64;
  if (global == "torch IntStorage")
    return DType::kI32;
  if (global == "torch ByteStorage")
    return DType::kU8;
  return std::nullopt;
}

std::vector<std::int64_t> Integers(const Ref& tuple, const char* what) {
  Require(tuple->kind == Object::Kind::kTuple,
          std::string("tensor ") + what + " is not a tuple");
  std::vector<std::int64_t> values;
  for (const Ref& item : tuple->items) {
    Require(item->kind == Object::Kind::kInt && item->integer >= 0,
            std::string("tensor ") + what + " holds a non-integer");
    values.push_back(item->integer);
  }
  return values;
}

/// A pickle interpreter for protocol-2 tensor state dicts.
class Unpickler {
public:
  explicit Unpickler(Bytes bytes) : bytes_(bytes) {}

  Ref Load() {
    while (true) {
      const std::uint8_t op = Byte();
      switch (op) {
        case 0x80:  // PROTO
          Require(Byte() <= 5, "unsupported pickle protocol");
          break;
        case '.':  // STOP
          Require(stack_.size() == 1, "pickle stack is unbalanced");
          return stack_.back();
        case '(':  // MARK
          marks_.push_back(stack_.size());
          break;
        case '}':  // EMPTY_DICT
          Push(Make(Object::Kind::kDict));
          break;
        case ']':  // EMPTY_LIST
          Push(Make(Object::Kind::kList));
          break;
        case ')':  // EMPTY_TUPLE
          Push(Make(Object::Kind::kTuple));
          break;
        case 'N':  // NONE
          Push(Make(Object::Kind::kNone));
          break;
        case 0x88:    // NEWTRUE
        case 0x89: {  // NEWFALSE
          Ref value = Make(Object::Kind::kBool);
          value->integer = op == 0x88 ? 1 : 0;
          Push(value);
          break;
        }
        case 'J':  // BININT
          Push(MakeInt(static_cast<std::int32_t>(Read(4))));
          break;
        case 'K':  // BININT1
          Push(MakeInt(static_cast<std::int64_t>(Read(1))));
          break;
        case 'M':  // BININT2
          Push(MakeInt(static_cast<std::int64_t>(Read(2))));
          break;
        case 0x8a: {  // LONG1
          const std::size_t length = Byte();
          Require(length <= 8, "pickle integer is too long");
          std::uint64_t value = Read(static_cast<int>(length));
          if (length > 0 && length < 8 && (value >> (8 * length - 1)) != 0)
            value |= ~std::uint64_t{0} << (8 * length);
          Push(MakeInt(static_cast<std::int64_t>(value)));
          break;
        }
        case 'G': {  // BINFLOAT (big endian)
          std::uint64_t bits = 0;
          for (int index = 0; index < 8; ++index)
            bits = (bits << 8U) | Byte();
          Ref value = Make(Object::Kind::kFloat);
          std::memcpy(&value->number, &bits, sizeof(bits));
          Push(value);
          break;
        }
        case 'X':  // BINUNICODE
          PushString(Read(4));
          break;
        case 0x8c:  // SHORT_BINUNICODE
          PushString(Byte());
          break;
        case 'q':  // BINPUT
          memo_[static_cast<std::uint32_t>(Read(1))] = Top();
          break;
        case 'r':  // LONG_BINPUT
          memo_[static_cast<std::uint32_t>(Read(4))] = Top();
          break;
        case 'h':  // BINGET
          Push(Memo(static_cast<std::uint32_t>(Read(1))));
          break;
        case 'j':  // LONG_BINGET
          Push(Memo(static_cast<std::uint32_t>(Read(4))));
          break;
        case 0x85:    // TUPLE1
        case 0x86:    // TUPLE2
        case 0x87: {  // TUPLE3
          const std::size_t count = op - 0x84;
          Require(stack_.size() >= count, "pickle stack underflow");
          Ref tuple = Make(Object::Kind::kTuple);
          tuple->items.assign(stack_.end() - static_cast<long>(count),
                              stack_.end());
          stack_.resize(stack_.size() - count);
          Push(tuple);
          break;
        }
        case 't': {  // TUPLE
          Ref tuple = Make(Object::Kind::kTuple);
          tuple->items = PopMark();
          Push(tuple);
          break;
        }
        case 'a': {  // APPEND
          Ref value = Pop();
          Ref list = Top();
          Require(list->kind == Object::Kind::kList, "APPEND to a non-list");
          list->items.push_back(value);
          break;
        }
        case 'e': {  // APPENDS
          std::vector<Ref> values = PopMark();
          Ref list = Top();
          Require(list->kind == Object::Kind::kList, "APPENDS to a non-list");
          list->items.insert(list->items.end(), values.begin(), values.end());
          break;
        }
        case 's': {  // SETITEM
          Ref value = Pop();
          Ref key = Pop();
          SetItem(Top(), key, value);
          break;
        }
        case 'u': {  // SETITEMS
          std::vector<Ref> values = PopMark();
          Require(values.size() % 2 == 0, "SETITEMS with an odd count");
          for (std::size_t index = 0; index < values.size(); index += 2)
            SetItem(Top(), values[index], values[index + 1]);
          break;
        }
        case 'c': {  // GLOBAL
          const std::string module = Line();
          const std::string name = Line();
          const std::string global = module + " " + name;
          Require(global == "collections OrderedDict" ||
                      global == "torch._utils _rebuild_tensor_v2" ||
                      StorageType(global).has_value(),
                  "checkpoint references " + global);
          Ref value = Make(Object::Kind::kGlobal);
          value->text = global;
          Push(value);
          break;
        }
        case 'R': {  // REDUCE
          Ref arguments = Pop();
          Ref callable = Pop();
          Push(Reduce(callable, arguments));
          break;
        }
        case 'Q': {  // BINPERSID
          Ref id = Pop();
          Push(PersistentLoad(id));
          break;
        }
        case 'b':  // BUILD: OrderedDict._metadata and similar; dropped.
          Pop();
          Require(!stack_.empty(), "BUILD without an object");
          break;
        default:
          throw FormatError("unsupported pickle opcode " + std::to_string(op));
      }
    }
  }

private:
  std::uint8_t Byte() {
    Require(position_ < bytes_.size, "pickle ends early");
    return bytes_.data[position_++];
  }

  std::uint64_t Read(int width) {
    const std::uint64_t value = bytes_.Read(position_, width);
    position_ += static_cast<std::size_t>(width);
    return value;
  }

  std::string Line() {
    std::string text;
    for (char c = static_cast<char>(Byte()); c != '\n';
         c = static_cast<char>(Byte()))
      text.push_back(c);
    return text;
  }

  void PushString(std::uint64_t length) {
    Require(position_ + length <= bytes_.size, "pickle string runs past");
    Ref value = Make(Object::Kind::kString);
    value->text.assign(reinterpret_cast<const char*>(bytes_.data + position_),
                       length);
    position_ += length;
    Push(value);
  }

  void Push(Ref value) { stack_.push_back(std::move(value)); }

  Ref Pop() {
    Require(
        !stack_.empty() && (marks_.empty() || stack_.size() > marks_.back()),
        "pickle stack underflow");
    Ref value = stack_.back();
    stack_.pop_back();
    return value;
  }

  const Ref& Top() const {
    Require(!stack_.empty(), "pickle stack underflow");
    return stack_.back();
  }

  std::vector<Ref> PopMark() {
    Require(!marks_.empty(), "pickle mark missing");
    const std::size_t mark = marks_.back();
    marks_.pop_back();
    std::vector<Ref> values(stack_.begin() + static_cast<long>(mark),
                            stack_.end());
    stack_.resize(mark);
    return values;
  }

  Ref Memo(std::uint32_t index) const {
    const auto found = memo_.find(index);
    Require(found != memo_.end(), "pickle memo entry missing");
    return found->second;
  }

  static void SetItem(const Ref& dict, Ref key, Ref value) {
    Require(dict->kind == Object::Kind::kDict, "SETITEM on a non-dict");
    dict->members.emplace_back(std::move(key), std::move(value));
  }

  static Ref Reduce(const Ref& callable, const Ref& arguments) {
    Require(callable->kind == Object::Kind::kGlobal &&
                arguments->kind == Object::Kind::kTuple,
            "REDUCE of a non-global");
    if (callable->text == "collections OrderedDict") {
      Require(arguments->items.empty(), "OrderedDict with arguments");
      return Make(Object::Kind::kDict);
    }
    Require(callable->text == "torch._utils _rebuild_tensor_v2",
            "REDUCE of " + callable->text);
    // (storage, storage_offset, size, stride, requires_grad,
    //  backward_hooks[, metadata])
    const std::vector<Ref>& items = arguments->items;
    Require(items.size() >= 4 && items[0]->kind == Object::Kind::kStorage &&
                items[1]->kind == Object::Kind::kInt,
            "malformed _rebuild_tensor_v2 call");
    Ref tensor = Make(Object::Kind::kTensor);
    tensor->storage = items[0];
    tensor->integer = items[1]->integer;
    tensor->size = Integers(items[2], "size");
    tensor->stride = Integers(items[3], "stride");
    Require(tensor->size.size() == tensor->stride.size(),
            "tensor size and stride differ in rank");
    return tensor;
  }

  static Ref PersistentLoad(const Ref& id) {
    // ("storage", storage type, key, location, element count)
    Require(id->kind == Object::Kind::kTuple && id->items.size() == 5 &&
                id->items[0]->kind == Object::Kind::kString &&
                id->items[0]->text == "storage" &&
                id->items[1]->kind == Object::Kind::kGlobal &&
                id->items[2]->kind == Object::Kind::kString &&
                id->items[4]->kind == Object::Kind::kInt,
            "unsupported persistent id");
    const auto dtype = StorageType(id->items[1]->text);
    Require(dtype.has_value(), "unsupported storage " + id->items[1]->text);
    Ref storage = Make(Object::Kind::kStorage);
    storage->dtype = *dtype;
    storage->text = id->items[2]->text;
    storage->count = static_cast<std::uint64_t>(id->items[4]->integer);
    return storage;
  }

  Bytes bytes_;
  std::size_t position_ = 0;
  std::vector<Ref> stack_;
  std::vector<std::size_t> marks_;
  std::unordered_map<std::uint32_t, Ref> memo_;
};

}  // namespace

TorchCheckpoint::~TorchCheckpoint() {
  mapping_.Close();
}

std::unique_ptr<TorchCheckpoint> TorchCheckpoint::Open(
    const std::filesystem::path& path, std::string* error) {
  std::unique_ptr<TorchCheckpoint> checkpoint(new TorchCheckpoint());
  std::string open_error;
  checkpoint->mapping_ = platform::ReadOnlyMappedFile::Open(path, &open_error);
  if (!checkpoint->mapping_.valid()) {
    if (error != nullptr)
      *error = "cannot map " + path.string() + ": " + open_error;
    return nullptr;
  }
  try {
    const Bytes file{checkpoint->mapping_.data(), checkpoint->mapping_.size()};
    const std::map<std::string, Entry> entries = ReadZip(file);
    std::string prefix;
    for (const auto& [name, entry] : entries) {
      const std::size_t slash = name.find('/');
      if (slash != std::string::npos && name.substr(slash + 1) == "data.pkl") {
        prefix = name.substr(0, slash + 1);
        break;
      }
    }
    Require(!prefix.empty(), "the archive lacks data.pkl");
    const auto order = entries.find(prefix + "byteorder");
    if (order != entries.end())
      Require(std::string_view(reinterpret_cast<const char*>(
                                   file.data + order->second.offset),
                               order->second.size) == "little",
              "the checkpoint is not little endian");
    const Entry& pickle = entries.at(prefix + "data.pkl");
    Ref root = Unpickler(Bytes{file.data + pickle.offset, pickle.size}).Load();
    Require(root->kind == Object::Kind::kDict, "the checkpoint is not a dict");
    for (const auto& [key, value] : root->members) {
      if (key->kind == Object::Kind::kString && key->text == "state_dict" &&
          value->kind == Object::Kind::kDict) {
        root = value;
        break;
      }
    }
    for (const auto& [key, value] : root->members) {
      Require(key->kind == Object::Kind::kString &&
                  value->kind == Object::Kind::kTensor,
              "the state dict holds a non-tensor entry");
      const Object& storage = *value->storage;
      const auto found = entries.find(prefix + "data/" + storage.text);
      Require(found != entries.end(),
              "storage " + storage.text + " of " + key->text + " is missing");
      const std::size_t element = DTypeSize(storage.dtype);
      Require(found->second.size >= storage.count * element,
              "storage " + storage.text + " is truncated");
      Tensor tensor;
      tensor.name = key->text;
      tensor.dtype = storage.dtype;
      std::uint64_t elements = 1;
      for (std::size_t dim = value->size.size(); dim-- > 0;) {
        const auto size = static_cast<std::uint64_t>(value->size[dim]);
        Require(size <= 1 ||
                    static_cast<std::uint64_t>(value->stride[dim]) == elements,
                key->text + " is not contiguous");
        elements *= size;
      }
      for (const std::int64_t size : value->size)
        tensor.shape.push_back(static_cast<std::size_t>(size));
      const auto offset = static_cast<std::uint64_t>(value->integer);
      Require(offset + elements <= storage.count,
              key->text + " lies outside its storage");
      tensor.offset = found->second.offset + offset * element;
      tensor.bytes = static_cast<std::size_t>(elements * element);
      tensor.data =
          reinterpret_cast<const std::byte*>(file.data + tensor.offset);
      checkpoint->tensors_.push_back(std::move(tensor));
    }
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = path.string() + ": " + failure.what();
    return nullptr;
  }
  return checkpoint;
}

const Tensor* TorchCheckpoint::Find(std::string_view name) const {
  for (const Tensor& tensor : tensors_)
    if (tensor.name == name)
      return &tensor;
  return nullptr;
}

}  // namespace gufo::models::voxcpm
