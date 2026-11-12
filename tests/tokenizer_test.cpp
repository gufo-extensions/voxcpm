#include "tokenizer.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace json = gufo::json;
namespace vc = gufo::models::voxcpm;

namespace {

constexpr int kSkip = 77;

std::string Describe(const std::vector<std::uint32_t>& ids) {
  std::string text = "[";
  for (std::size_t index = 0; index < ids.size(); ++index)
    text += (index == 0 ? "" : ", ") + std::to_string(ids[index]);
  return text + "]";
}

}  // namespace

int main() {
  const char* root = std::getenv("GUFO_VOXCPM_MODEL");
  if (root == nullptr || *root == '\0') {
    std::cout << "skip: GUFO_VOXCPM_MODEL is not set\n";
    return kSkip;
  }
  std::string error;
  const auto tokenizer = vc::Tokenizer::Load(
      std::filesystem::path(root) / "tokenizer.json", &error);
  if (tokenizer == nullptr) {
    std::cout << "FAIL load: " << error << '\n';
    return 1;
  }
  int failures = 0;
  if (tokenizer->vocabulary_size() != 73448 ||
      tokenizer->TokenId("<|im_end|>") != 73440u ||
      tokenizer->TokenId("<s>") != 1u) {
    std::cout << "FAIL vocabulary size or added token IDs\n";
    ++failures;
  }
  std::ifstream stream(GUFO_VOXCPM_TOKENIZER_GOLDEN);
  const json::Value golden =
      json::parse(std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>()));
  std::size_t cases = 0;
  for (const json::Value& entry : golden.find("cases")->items()) {
    std::vector<std::uint32_t> expected;
    for (const json::Value& id : entry.find("ids")->items())
      expected.push_back(static_cast<std::uint32_t>(id.as_size()));
    const std::string& text = entry.find("text")->str();
    std::vector<std::uint32_t> ids;
    if (!tokenizer->Encode(text, &ids, &error)) {
      std::cout << "FAIL \"" << text << "\": " << error << '\n';
      ++failures;
    } else if (ids != expected) {
      std::cout << "FAIL \"" << text << "\"\n  got      " << Describe(ids)
                << "\n  expected " << Describe(expected) << '\n';
      ++failures;
    }
    ++cases;
  }
  std::vector<std::uint32_t> ids;
  if (tokenizer->Encode("bad \xC3\x28 byte", &ids, &error)) {
    std::cout << "FAIL invalid UTF-8 accepted\n";
    ++failures;
  }
  std::cout << (failures == 0 ? "ok   " : "FAIL ") << cases << " golden cases, "
            << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
