#ifndef GUFO_MODELS_VOXCPM_TOKENIZER_HPP_
#define GUFO_MODELS_VOXCPM_TOKENIZER_HPP_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::voxcpm {

/// VoxCPM2's text tokenizer: the SentencePiece-style BPE of `tokenizer.json`
/// (normalizer: prepend "▁", spaces to "▁"; no pre-tokenizer; byte
/// fallback) followed by VoxCPM's `mask_multichar_chinese_tokens`, which
/// replaces every token made of two or more CJK ideographs (U+4E00..U+9FFF)
/// by the tokens of its single characters. Added tokens (`<s>`,
/// `<|im_end|>`, ...) in the text are matched before normalization, as the
/// `tokenizers` library does. No BOS is added.
class Tokenizer {
public:
  [[nodiscard]] static std::unique_ptr<Tokenizer> Load(
      const std::filesystem::path& path, std::string* error = nullptr);
  /// Parses the text of a tokenizer.json.
  [[nodiscard]] static std::unique_ptr<Tokenizer> Parse(
      std::string_view text, std::string* error = nullptr);

  /// Appends the IDs of `text` (UTF-8) to `ids`; fails on invalid UTF-8.
  [[nodiscard]] bool Encode(std::string_view text,
                            std::vector<std::uint32_t>* ids,
                            std::string* error = nullptr) const;
  /// The ID of a vocabulary or added token.
  [[nodiscard]] std::optional<std::uint32_t> TokenId(
      std::string_view token) const;
  /// One past the largest ID.
  [[nodiscard]] std::size_t vocabulary_size() const { return size_; }

private:
  struct Merge {
    std::uint32_t rank = 0;
    std::uint32_t id = 0;
  };
  struct Added {
    std::string content;
    std::uint32_t id = 0;
  };

  Tokenizer() = default;
  void EncodeSegment(std::string_view segment,
                     std::vector<std::uint32_t>* ids) const;

  std::unordered_map<std::string, std::uint32_t> vocab_;
  // Key: left ID << 32 | right ID.
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::vector<Added> added_;
  std::uint32_t byte_ids_[256] = {};
  std::uint32_t unk_id_ = 0;
  // Multi-character CJK tokens and the IDs of their characters.
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> splits_;
  std::size_t size_ = 0;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_TOKENIZER_HPP_
