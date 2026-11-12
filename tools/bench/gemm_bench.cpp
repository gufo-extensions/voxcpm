// Times the product kernels on VoxCPM2's shapes and reports the weight
// bandwidth (bytes of W per second): plain reads, the bf16 kernels
// (SkinnyGemm variants; matrix cores on bf16 tiles) and the matrix cores on
// Q8_0 tiles, each with several wave counts.
//
//   voxcpm_gemm_bench [REPEATS [q8]]   (q8: only the Q8_0 tiles)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "device.hpp"
#include "gemm.hpp"
#include "kernels/skinny_gemm.hpp"
#include "kernels/wmma_products.hpp"

namespace vc = gufo::models::voxcpm;
namespace kernels = gufo::models::voxcpm::kernels;

namespace {

struct Shape {
  const char* name;
  std::size_t in;
  std::size_t out;
};

constexpr Shape kShapes[] = {
    {"lm qkv", 2048, 2560},        {"lm o", 2048, 2048},
    {"lm gate_up", 2048, 12288},   {"lm down", 6144, 2048},
    {"local qkv", 1024, 2560},     {"local o", 2048, 1024},
    {"local gate_up", 1024, 8192}, {"local down", 4096, 1024},
};

// Distinct weights per repeat, at least kRotation bytes per shape, so that
// the products read DRAM like the model does, not Strix Halo's 32 MB MALL.
constexpr double kRotation = 128e6;
constexpr int kMinCopies = 8;

int Copies(double bytes) {
  return std::max(kMinCopies, static_cast<int>(std::ceil(kRotation / bytes)));
}

class Timer {
public:
  Timer(hipStream_t stream, int repeats) : stream_(stream), repeats_(repeats) {
    vc::Check(hipEventCreate(&start_), "event");
    vc::Check(hipEventCreate(&stop_), "event");
  }
  ~Timer() {
    (void)hipEventDestroy(start_);
    (void)hipEventDestroy(stop_);
  }

  /// Seconds per call of `run(copy)` over `copies` copies; negative when a
  /// call launched nothing (false).
  double Time(int copies, const std::function<bool(int)>& run) {
    bool launched = true;
    for (int i = 0; i < 3; ++i)
      launched = run(i % copies) && launched;
    if (!launched)
      return -1.0;
    vc::Check(hipEventRecord(start_, stream_), "record");
    for (int i = 0; i < repeats_; ++i)
      (void)run(i % copies);
    vc::Check(hipEventRecord(stop_, stream_), "record");
    vc::Check(hipEventSynchronize(stop_), "sync");
    float ms = 0.0F;
    vc::Check(hipEventElapsedTime(&ms, start_, stop_), "elapsed");
    return ms / 1e3 / repeats_;
  }

private:
  hipStream_t stream_;
  int repeats_;
  hipEvent_t start_ = nullptr;
  hipEvent_t stop_ = nullptr;
};

void Report(const std::string& label, double seconds, double bytes) {
  if (seconds < 0.0)
    std::printf("  %s %7s    %6s     ", label.c_str(), "n/a", "");
  else
    std::printf("  %s %7.1f us %6.1f GB/s", label.c_str(), seconds * 1e6,
                bytes / seconds / 1e9);
}

}  // namespace

int main(int argc, char** argv) {
  const int repeats = argc > 1 ? std::atoi(argv[1]) : 50;
  const bool q8_only = argc > 2 && std::string(argv[2]) == "q8";
  hipStream_t stream = nullptr;
  vc::Check(hipStreamCreate(&stream), "stream");
  // One pool per weight layout, large enough for every shape's copies.
  std::size_t pool = 0;
  for (const Shape& shape : kShapes) {
    const double bytes = 2.0 * shape.in * shape.out;
    pool = std::max(pool, static_cast<std::size_t>(Copies(bytes) * bytes));
  }
  vc::DeviceBuffer<std::uint8_t> weights(pool);
  vc::DeviceBuffer<std::uint8_t> bf16_tiles(pool);
  vc::DeviceBuffer<std::uint8_t> q8_tiles(pool);
  vc::DeviceBuffer<std::uint16_t> x(32 * 8192);
  vc::DeviceBuffer<float> y(32 * 12288);
  vc::DeviceBuffer<float> partials(64 * 32 * 12288);
  vc::DeviceBuffer<unsigned> counters(12288 / 16);
  vc::Check(hipMemset(counters.get(), 0, counters.size() * sizeof(unsigned)),
            "counters");
  const kernels::SplitScratch scratch{partials.get(), partials.size(),
                                      counters.get(), counters.size()};
  {
    std::vector<std::uint16_t> host(pool / 2);
    for (std::size_t i = 0; i < host.size(); ++i)
      host[i] = static_cast<std::uint16_t>(0x3C00 + (i * 7919) % 512);
    vc::Check(
        hipMemcpy(weights.get(), host.data(), pool, hipMemcpyHostToDevice),
        "upload");
    std::vector<std::uint16_t> activations(32 * 8192);
    for (std::size_t i = 0; i < activations.size(); ++i)
      activations[i] = static_cast<std::uint16_t>(0x3E00 + (i * 31) % 256);
    vc::Check(hipMemcpy(x.get(), activations.data(),
                        activations.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice),
              "x");
  }
  const auto* bf16 = reinterpret_cast<const kernels::Bf16*>(weights.get());
  vc::DeviceBuffer<unsigned> sink(1);
  Timer timer(stream, repeats);

  std::printf("plain reads\n");
  for (const Shape& shape : kShapes) {
    if (q8_only)
      break;
    const double bytes = 2.0 * shape.in * shape.out;
    const int copies = Copies(bytes);
    const std::size_t count = shape.in * shape.out / 8;
    std::printf("read     %-14s", shape.name);
    for (const unsigned blocks : {320U, 1280U, 5120U}) {
      const double seconds = timer.Time(copies, [&](int copy) {
        kernels::ReadProbe(bf16 + copy * shape.in * shape.out, count, blocks,
                           sink.get(), stream);
        return true;
      });
      Report("b" + std::to_string(blocks), seconds, bytes);
    }
    std::printf("\n");
  }

  std::printf(
      "bf16 weights: wave per row (v1, v4), sweep (v0), matrix "
      "cores on tiles (w<target waves>)\n");
  for (const std::size_t rows :
       {std::size_t{1}, std::size_t{5}, std::size_t{22}}) {
    if (q8_only)
      break;
    for (const Shape& shape : kShapes) {
      const std::size_t elements = shape.in * shape.out;
      const double bytes = 2.0 * elements;
      const int copies = Copies(bytes);
      std::printf("rows %2zu %-14s", rows, shape.name);
      for (const int vec : {1, 4, 0}) {
        const double seconds = timer.Time(copies, [&](int copy) {
          return kernels::SkinnyGemmWith(vec, x.get(), bf16 + copy * elements,
                                         y.get(), rows, shape.in, shape.out,
                                         false, stream);
        });
        Report("v" + std::to_string(vec), seconds, bytes);
      }
      if (rows > 8) {
        auto* tiles = reinterpret_cast<kernels::Bf16*>(bf16_tiles.get());
        for (int copy = 0; copy < copies; ++copy)
          kernels::Bf16ToTiles(bf16 + copy * elements, shape.out, shape.in,
                               tiles + copy * elements, stream);
        for (const std::size_t waves : {160, 320, 640, 1280}) {
          const kernels::WmmaPlan plan =
              kernels::PlanWmma(rows, shape.in, shape.out, waves);
          const double seconds = timer.Time(copies, [&](int copy) {
            return kernels::WmmaBf16(x.get(), tiles + copy * elements, y.get(),
                                     rows, shape.in, shape.out, false, plan,
                                     scratch, stream);
          });
          Report("w" + std::to_string(waves), seconds, bytes);
        }
      }
      std::printf("\n");
    }
  }

  std::printf("Q8_0 tiles: matrix cores (w<target waves>)\n");
  for (const Shape& shape : kShapes) {
    // Random GGML blocks, repacked per copy.
    const std::size_t blocks = shape.in * shape.out / 32;
    std::vector<std::uint8_t> host(blocks * 34);
    std::mt19937 rng(7);
    for (std::size_t block = 0; block < blocks; ++block) {
      host[block * 34] = 0x00;
      host[block * 34 + 1] = 0x14;  // scale 2^-10
      for (int i = 0; i < 32; ++i)
        host[block * 34 + 2 + i] = static_cast<std::uint8_t>(rng());
    }
    vc::DeviceBuffer<std::uint8_t> raw(host.size());
    vc::Check(
        hipMemcpy(raw.get(), host.data(), host.size(), hipMemcpyHostToDevice),
        "upload");
    const std::size_t tile_bytes = kernels::Q8TilesBytes(shape.out, shape.in);
    const int copies = Copies(static_cast<double>(tile_bytes));
    for (int copy = 0; copy < copies; ++copy)
      kernels::Q8_0ToTiles(raw.get(), shape.out, shape.in,
                           q8_tiles.get() + copy * tile_bytes, stream);
    vc::Check(hipStreamSynchronize(stream), "sync");
    for (const std::size_t rows :
         {std::size_t{1}, std::size_t{5}, std::size_t{22}}) {
      std::printf("rows %2zu %-14s", rows, shape.name);
      for (const std::size_t waves : {160, 320, 640, 1280}) {
        const kernels::WmmaPlan plan =
            kernels::PlanWmma(rows, shape.in, shape.out, waves);
        const double seconds = timer.Time(copies, [&](int copy) {
          return kernels::WmmaQ8(x.get(), q8_tiles.get() + copy * tile_bytes,
                                 y.get(), rows, shape.in, shape.out, false,
                                 plan, scratch, stream);
        });
        Report("w" + std::to_string(waves), seconds,
               static_cast<double>(tile_bytes));
      }
      std::printf("\n");
    }
  }
  return 0;
}
