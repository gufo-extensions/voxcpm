// The multi-character Chinese token split follows OpenBMB/VoxCPM's
// mask_multichar_chinese_tokens (src/voxcpm/model/utils.py; Apache-2.0,
// Copyright OpenBMB; see licenses/VoxCPM.txt), rewritten in C++.

#include "tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <utility>

namespace gufo::models::voxcpm {
namespace {

namespace json = gufo::json;

constexpr std::string_view kSpace = "\xE2\x96\x81";  // U+2581

/// Length of the UTF-8 sequence at `text[offset]`, 0 if it is invalid.
std::size_t SequenceLength(std::string_view text, std::size_t offset) {
  const auto lead = static_cast<unsigned char>(text[offset]);
  const std::size_t length = lead < 0x80                    ? 1
                             : lead >= 0xC2 && lead <= 0xDF ? 2
                             : lead >= 0xE0 && lead <= 0xEF ? 3
                             : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                            : 0;
  if (length == 0 || offset + length > text.size())
    return 0;
  for (std::size_t index = 1; index < length; ++index) {
    const auto byte = static_cast<unsigned char>(text[offset + index]);
    if (byte < 0x80 || byte > 0xBF)
      return 0;
    if (index == 1 &&
        ((lead == 0xE0 && byte < 0xA0) || (lead == 0xED && byte > 0x9F) ||
         (lead == 0xF0 && byte < 0x90) || (lead == 0xF4 && byte > 0x8F)))
      return 0;
  }
  return length;
}

/// The code point of a valid sequence.
char32_t Decode(std::string_view sequence) {
  const auto lead = static_cast<unsigned char>(sequence[0]);
  if (sequence.size() == 1)
    return lead;
  char32_t value = lead & (0x7F >> sequence.size());
  for (std::size_t index = 1; index < sequence.size(); ++index)
    value = (value << 6) | (static_cast<unsigned char>(sequence[index]) & 0x3F);
  return value;
}

/// The code-point pieces of valid UTF-8, or nullopt.
std::optional<std::vector<std::string_view>> Characters(std::string_view text) {
  std::vector<std::string_view> characters;
  for (std::size_t offset = 0; offset < text.size();) {
    const std::size_t length = SequenceLength(text, offset);
    if (length == 0)
      return std::nullopt;
    characters.push_back(text.substr(offset, length));
    offset += length;
  }
  return characters;
}

bool IsHan(char32_t c) {
  return c >= 0x4E00 && c <= 0x9FFF;
}

std::uint64_t PairKey(std::uint32_t left, std::uint32_t right) {
  return (static_cast<std::uint64_t>(left) << 32) | right;
}

class LoadError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw LoadError("tokenizer.json: " + message);
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Load(const std::filesystem::path& path,
                                           std::string* error) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    if (error != nullptr)
      *error = "cannot read " + path.string();
    return nullptr;
  }
  return Parse(std::string(std::istreambuf_iterator<char>(stream),
                           std::istreambuf_iterator<char>()),
               error);
}

std::unique_ptr<Tokenizer> Tokenizer::Parse(std::string_view text,
                                            std::string* error) {
  std::unique_ptr<Tokenizer> tokenizer(new Tokenizer());
  try {
    const json::Value root = json::parse(text);
    const json::Value* model = root.find("model");
    Require(model != nullptr && model->is_object(), "lacks the model");
    const json::Value* type = model->find("type");
    Require(type != nullptr && type->is_string() && type->str() == "BPE",
            "the model is not BPE");
    const json::Value* fallback = model->find("byte_fallback");
    Require(fallback != nullptr && fallback->is_bool() && fallback->as_bool(),
            "byte fallback is off");

    // Only the normalizer VoxCPM2 ships: Prepend("▁") then Replace(" ",
    // "▁"), with no pre-tokenizer.
    const json::Value* pre = root.find("pre_tokenizer");
    Require(pre == nullptr || pre->is_null(), "has a pre-tokenizer");
    const json::Value* normalizer = root.find("normalizer");
    Require(normalizer != nullptr && normalizer->is_object(),
            "lacks the normalizer");
    const json::Value* steps = normalizer->find("normalizers");
    Require(steps != nullptr && steps->is_array() && steps->size() == 2,
            "has an unexpected normalizer");
    const json::Value& prepend = steps->items()[0];
    const json::Value& replace = steps->items()[1];
    Require(prepend.find("type")->str() == "Prepend" &&
                prepend.find("prepend")->str() == kSpace &&
                replace.find("type")->str() == "Replace" &&
                replace.find("pattern")->find("String")->str() == " " &&
                replace.find("content")->str() == kSpace,
            "has an unexpected normalizer");

    const json::Value* vocab = model->find("vocab");
    Require(vocab != nullptr && vocab->is_object(), "lacks the vocabulary");
    for (const auto& [token, id] : vocab->members()) {
      const auto value = static_cast<std::uint32_t>(id.as_size());
      tokenizer->vocab_.emplace(token, value);
      tokenizer->size_ =
          std::max(tokenizer->size_, static_cast<std::size_t>(value) + 1);
    }
    const json::Value* unk = model->find("unk_token");
    Require(unk != nullptr && unk->is_string(), "lacks the unknown token");
    const auto unk_id = tokenizer->vocab_.find(unk->str());
    Require(unk_id != tokenizer->vocab_.end(), "the unknown token is unknown");
    tokenizer->unk_id_ = unk_id->second;
    for (int byte = 0; byte < 256; ++byte) {
      char name[8];
      std::snprintf(name, sizeof(name), "<0x%02X>", byte);
      const auto found = tokenizer->vocab_.find(name);
      Require(found != tokenizer->vocab_.end(),
              std::string("lacks the byte token ") + name);
      tokenizer->byte_ids_[byte] = found->second;
    }

    const json::Value* merges = model->find("merges");
    Require(merges != nullptr && merges->is_array(), "lacks the merges");
    std::uint32_t rank = 0;
    for (const json::Value& merge : merges->items()) {
      std::string left, right;
      if (merge.is_string()) {
        const std::string& pair = merge.str();
        const std::size_t space = pair.find(' ');
        Require(space != std::string::npos, "has a malformed merge");
        left = pair.substr(0, space);
        right = pair.substr(space + 1);
      } else {
        Require(merge.is_array() && merge.size() == 2, "has a malformed merge");
        left = merge.items()[0].str();
        right = merge.items()[1].str();
      }
      const auto a = tokenizer->vocab_.find(left);
      const auto b = tokenizer->vocab_.find(right);
      const auto merged = tokenizer->vocab_.find(left + right);
      Require(a != tokenizer->vocab_.end() && b != tokenizer->vocab_.end() &&
                  merged != tokenizer->vocab_.end(),
              "merges tokens outside the vocabulary");
      tokenizer->merges_.emplace(PairKey(a->second, b->second),
                                 Merge{rank, merged->second});
      ++rank;
    }

    const json::Value* added = root.find("added_tokens");
    if (added != nullptr && added->is_array()) {
      for (const json::Value& token : added->items()) {
        const auto flag = [&token](const char* key) {
          const json::Value* value = token.find(key);
          return value != nullptr && value->is_bool() && value->as_bool();
        };
        Require(!flag("single_word") && !flag("lstrip") && !flag("rstrip") &&
                    !flag("normalized"),
                "has an added token with matching options");
        const auto id = static_cast<std::uint32_t>(token.find("id")->as_size());
        tokenizer->added_.push_back({token.find("content")->str(), id});
        tokenizer->size_ =
            std::max(tokenizer->size_, static_cast<std::size_t>(id) + 1);
      }
    }
    // Leftmost-longest matching tries longer contents first.
    std::stable_sort(tokenizer->added_.begin(), tokenizer->added_.end(),
                     [](const Added& a, const Added& b) {
                       return a.content.size() > b.content.size();
                     });

    // mask_multichar_chinese_tokens: the reference removes every "▁" from a
    // produced token and splits it into characters when the rest is a
    // vocabulary token of two or more Han characters.
    for (const auto& [token, id] : tokenizer->vocab_) {
      std::string clean;
      for (std::size_t offset = 0; offset < token.size();) {
        if (token.compare(offset, kSpace.size(), kSpace) == 0) {
          offset += kSpace.size();
        } else {
          clean.push_back(token[offset++]);
        }
      }
      const auto characters = Characters(clean);
      if (!characters.has_value() || characters->size() < 2 ||
          !tokenizer->vocab_.contains(clean))
        continue;
      std::vector<std::uint32_t> split;
      bool han = true;
      for (const std::string_view character : *characters) {
        if (!IsHan(Decode(character))) {
          han = false;
          break;
        }
        const auto found = tokenizer->vocab_.find(std::string(character));
        split.push_back(found == tokenizer->vocab_.end() ? tokenizer->unk_id_
                                                         : found->second);
      }
      if (han)
        tokenizer->splits_.emplace(id, std::move(split));
    }
  } catch (const std::exception& failure) {
    if (error != nullptr)
      *error = failure.what();
    return nullptr;
  }
  return tokenizer;
}

std::optional<std::uint32_t> Tokenizer::TokenId(std::string_view token) const {
  const auto found = vocab_.find(std::string(token));
  if (found != vocab_.end())
    return found->second;
  for (const Added& added : added_)
    if (added.content == token)
      return added.id;
  return std::nullopt;
}

bool Tokenizer::Encode(std::string_view text, std::vector<std::uint32_t>* ids,
                       std::string* error) const {
  if (!Characters(text).has_value()) {
    if (error != nullptr)
      *error = "the text is not valid UTF-8";
    return false;
  }
  std::size_t start = 0;
  for (std::size_t offset = 0; offset < text.size();) {
    const Added* match = nullptr;
    for (const Added& added : added_) {
      if (text.substr(offset, added.content.size()) == added.content) {
        match = &added;
        break;
      }
    }
    if (match == nullptr) {
      offset += SequenceLength(text, offset);
      continue;
    }
    EncodeSegment(text.substr(start, offset - start), ids);
    ids->push_back(match->id);
    offset += match->content.size();
    start = offset;
  }
  EncodeSegment(text.substr(start), ids);
  return true;
}

void Tokenizer::EncodeSegment(std::string_view segment,
                              std::vector<std::uint32_t>* ids) const {
  if (segment.empty())
    return;
  std::string normalized(kSpace);
  for (const char c : segment) {
    if (c == ' ')
      normalized.append(kSpace);
    else
      normalized.push_back(c);
  }

  // Symbols of the word, linked in order; merged-away ones are dead.
  struct Symbol {
    std::uint32_t id;
    int prev;
    int next;
    bool alive;
  };
  std::vector<Symbol> symbols;
  const auto add = [&symbols](std::uint32_t id) {
    const int index = static_cast<int>(symbols.size());
    symbols.push_back({id, index - 1, index + 1, true});
  };
  const auto characters = *Characters(normalized);
  for (const std::string_view character : characters) {
    const auto found = vocab_.find(std::string(character));
    if (found != vocab_.end()) {
      add(found->second);
      continue;
    }
    for (const char byte : character)
      add(byte_ids_[static_cast<unsigned char>(byte)]);
  }
  symbols.back().next = -1;

  // The tokenizers library's Word::merge_all: lowest rank first, then the
  // leftmost position; stale entries are those whose pair no longer yields
  // the recorded token.
  struct Candidate {
    std::uint32_t rank;
    int position;
    std::uint32_t id;
    bool operator>(const Candidate& other) const {
      return rank != other.rank ? rank > other.rank : position > other.position;
    }
  };
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
  const auto consider = [&](int left) {
    if (left < 0 || symbols[left].next < 0)
      return;
    const auto found =
        merges_.find(PairKey(symbols[left].id, symbols[symbols[left].next].id));
    if (found != merges_.end())
      queue.push({found->second.rank, left, found->second.id});
  };
  for (int index = 0; index + 1 < static_cast<int>(symbols.size()); ++index)
    consider(index);
  while (!queue.empty()) {
    const Candidate top = queue.top();
    queue.pop();
    Symbol& left = symbols[top.position];
    if (!left.alive || left.next < 0)
      continue;
    const auto found = merges_.find(PairKey(left.id, symbols[left.next].id));
    if (found == merges_.end() || found->second.id != top.id)
      continue;
    Symbol& right = symbols[left.next];
    left.id = top.id;
    right.alive = false;
    left.next = right.next;
    if (right.next >= 0)
      symbols[right.next].prev = top.position;
    consider(left.prev);
    consider(top.position);
  }

  for (int index = 0; index >= 0; index = symbols[index].next) {
    const auto split = splits_.find(symbols[index].id);
    if (split == splits_.end())
      ids->push_back(symbols[index].id);
    else
      ids->insert(ids->end(), split->second.begin(), split->second.end());
  }
}

}  // namespace gufo::models::voxcpm
