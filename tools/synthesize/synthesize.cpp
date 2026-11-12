// Development driver: synthesizes one text to a 16-bit mono WAV file.
//
//   voxcpm_synthesize MODEL_DIR OUTPUT.wav TEXT [--instruct TEXT]
//       [--reference AUDIO [--reference-text TEXT]] [--seed N] [--cfg G]
//       [--timesteps N] [--repeat N] [--stream]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gufo/ext/audio.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "synthesizer.hpp"

namespace vc = gufo::models::voxcpm;

namespace {

void WriteLittleEndian(std::ofstream& stream, std::uint32_t value, int bytes) {
  for (int index = 0; index < bytes; ++index)
    stream.put(static_cast<char>((value >> (8 * index)) & 0xFFU));
}

bool WriteWav(const std::string& path, const std::vector<float>& samples,
              std::uint32_t rate) {
  std::ofstream stream(path, std::ios::binary);
  const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
  stream.write("RIFF", 4);
  WriteLittleEndian(stream, 36 + data_bytes, 4);
  stream.write("WAVEfmt ", 8);
  WriteLittleEndian(stream, 16, 4);
  WriteLittleEndian(stream, 1, 2);
  WriteLittleEndian(stream, 1, 2);
  WriteLittleEndian(stream, rate, 4);
  WriteLittleEndian(stream, rate * 2, 4);
  WriteLittleEndian(stream, 2, 2);
  WriteLittleEndian(stream, 16, 2);
  stream.write("data", 4);
  WriteLittleEndian(stream, data_bytes, 4);
  for (const float sample : samples) {
    const float clamped = std::clamp(sample, -1.0F, 1.0F);
    WriteLittleEndian(stream,
                      static_cast<std::uint32_t>(static_cast<std::int16_t>(
                          std::lround(clamped * 32767.0F))),
                      2);
  }
  return static_cast<bool>(stream);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: voxcpm_synthesize MODEL_DIR OUTPUT.wav TEXT "
                 "[--instruct TEXT] [--reference AUDIO [--reference-text "
                 "TEXT]] [--seed N] [--cfg G] [--timesteps N] [--repeat N] "
                 "[--stream]\n";
    return 2;
  }
  vc::SynthesisOptions options;
  options.text = argv[3];
  std::string reference;
  std::string reference_text;
  int repeat = 1;
  bool stream = false;
  for (int index = 4; index < argc; ++index) {
    const std::string_view flag = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc) {
        std::cerr << flag << " needs a value\n";
        std::exit(2);
      }
      return argv[++index];
    };
    if (flag == "--instruct")
      options.instruct = value();
    else if (flag == "--reference")
      reference = value();
    else if (flag == "--reference-text")
      reference_text = value();
    else if (flag == "--seed")
      options.seed = std::stoull(value());
    else if (flag == "--cfg")
      options.cfg = std::stof(value());
    else if (flag == "--timesteps")
      options.timesteps = std::stoul(value());
    else if (flag == "--repeat")
      repeat = std::stoi(value());
    else if (flag == "--stream")
      stream = true;
    else {
      std::cerr << "unknown option " << flag << '\n';
      return 2;
    }
  }

  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  const auto synthesizer = vc::Synthesizer::Load(argv[1], &error);
  if (synthesizer == nullptr) {
    std::cerr << "load failed: " << error << '\n';
    return 1;
  }
  std::cerr << "loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_start)
                   .count()
            << " s\n";
  vc::ReferenceVoice voice;
  if (!reference.empty()) {
    std::ifstream file(reference, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    gufo::core::AudioLimits limits;
    limits.formats = gufo::core::AudioFormats::kAll;
    limits.max_seconds = 120;
    limits.max_pcm_bits = 32;
    const gufo::core::DecodedAudio audio = gufo::core::DecodeAudio(
        std::as_bytes(std::span<const char>(bytes.data(), bytes.size())),
        limits);
    const auto start = std::chrono::steady_clock::now();
    if (!synthesizer->CloneVoice(audio.samples, audio.channels,
                                 audio.sample_rate, reference_text, &voice,
                                 &error)) {
      std::cerr << "reference failed: " << error << '\n';
      return 1;
    }
    std::cerr << "reference encoded in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << " s\n";
    options.voice = &voice;
  }

  std::vector<float> audio;
  for (int run = 0; run < repeat; ++run) {
    audio.clear();
    const auto start = std::chrono::steady_clock::now();
    double first = 0.0;
    std::size_t streamed = 0;
    vc::Synthesizer::AudioSink sink;
    if (stream) {
      sink = [&](std::span<const float> samples) {
        if (streamed == 0)
          first = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        streamed += samples.size();
        audio.insert(audio.end(), samples.begin(), samples.end());
        return true;
      };
    }
    if (!synthesizer->Synthesize(options, {}, sink, &audio, &error)) {
      std::cerr << "synthesis failed: " << error << '\n';
      return 1;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    const double length = static_cast<double>(audio.size()) /
                          static_cast<double>(synthesizer->SampleRate());
    std::cerr << "run " << run << ": " << length << " s audio in " << seconds
              << " s (RTF " << seconds / std::max(length, 1e-9) << ")";
    if (stream)
      std::cerr << ", first audio after " << first << " s";
    std::cerr << '\n';
  }
  if (!WriteWav(argv[2], audio, synthesizer->SampleRate())) {
    std::cerr << "cannot write " << argv[2] << '\n';
    return 1;
  }
  return 0;
}
