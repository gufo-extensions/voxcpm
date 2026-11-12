// The matrix-core products and the weight tiles against the host: random
// GGML Q8_0 blocks and bf16 activations at VoxCPM2's shapes and odd ones,
// every row count up to 32, products compared with a double-precision sum
// of the same operands; and Bf16Gemm::Linear on tiles, which chunks up to
// 128 rows and expands the tiles for hipBLASLt beyond.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "device.hpp"
#include "gemm.hpp"
#include "kernels/wmma_products.hpp"
#include "matrix.hpp"

namespace vc = gufo::models::voxcpm;
namespace kernels = gufo::models::voxcpm::kernels;

namespace {

constexpr int kSkip = 77;
int failures = 0;

float Bf16ToFloat(std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint16_t FloatToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >>
                                    16U);
}

float HalfToFloat(std::uint16_t bits) {
  const std::uint32_t sign = (bits >> 15U) & 1U;
  const std::uint32_t exponent = (bits >> 10U) & 0x1FU;
  const std::uint32_t mantissa = bits & 0x3FFU;
  float value = 0.0F;
  if (exponent == 0)
    value = std::ldexp(static_cast<float>(mantissa), -24);
  else
    value = std::ldexp(static_cast<float>(mantissa | 0x400U),
                       static_cast<int>(exponent) - 25);
  return sign != 0 ? -value : value;
}

/// Random Q8_0 weights [out][in] as GGML blocks, and their values.
struct Q8Weights {
  std::vector<std::uint8_t> blocks;
  std::vector<double> values;
};

Q8Weights MakeWeights(std::size_t out, std::size_t in, std::mt19937& rng) {
  Q8Weights weights;
  std::uniform_int_distribution<int> quant(-128, 127);
  // Scales around 1e-3, as trained weights have them (float16 normals).
  std::uniform_int_distribution<int> scale_bits(0x1000, 0x1C00);
  const std::size_t blocks = out * in / 32;
  weights.blocks.resize(blocks * 34);
  weights.values.resize(out * in);
  for (std::size_t block = 0; block < blocks; ++block) {
    std::uint8_t* target = weights.blocks.data() + block * 34;
    const auto bits = static_cast<std::uint16_t>(scale_bits(rng));
    target[0] = static_cast<std::uint8_t>(bits & 0xFFU);
    target[1] = static_cast<std::uint8_t>(bits >> 8U);
    const double scale = HalfToFloat(bits);
    for (std::size_t i = 0; i < 32; ++i) {
      const int q = quant(rng);
      target[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
      weights.values[block * 32 + i] = scale * q;
    }
  }
  return weights;
}

std::vector<std::uint16_t> MakeActivations(std::size_t rows, std::size_t in,
                                           std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0F, 2.0F);
  std::vector<std::uint16_t> x(rows * in);
  for (std::uint16_t& value : x)
    value = FloatToBf16(normal(rng));
  return x;
}

template<typename T>
vc::DeviceBuffer<T> Upload(const std::vector<T>& host) {
  vc::DeviceBuffer<T> device(host.size());
  vc::Check(hipMemcpy(device.get(), host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice),
            "upload");
  return device;
}

std::vector<float> Download(const vc::DeviceBuffer<float>& device,
                            std::size_t count) {
  std::vector<float> host(count);
  vc::Check(hipMemcpy(host.data(), device.get(), count * sizeof(float),
                      hipMemcpyDeviceToHost),
            "download");
  return host;
}

/// Compares y with the exact product; the limit scales with sum |x||w| of
/// each output (float32 sums of exact products).
void Compare(const std::string& name, const std::vector<float>& y,
             const std::vector<std::uint16_t>& x, const std::vector<double>& w,
             const std::vector<float>& before, std::size_t rows, std::size_t in,
             std::size_t out, double relative) {
  double worst = 0.0;
  for (std::size_t m = 0; m < rows; ++m) {
    for (std::size_t n = 0; n < out; ++n) {
      double exact = before.empty() ? 0.0 : before[m * out + n];
      double magnitude = 0.0;
      for (std::size_t k = 0; k < in; ++k) {
        const double product = Bf16ToFloat(x[m * in + k]) * w[n * in + k];
        exact += product;
        magnitude += std::abs(product);
      }
      const double error =
          std::abs(y[m * out + n] - exact) / std::max(magnitude, 1e-30);
      worst = std::max(worst, error);
    }
  }
  const bool ok = worst <= relative;
  // hipBLASLt leaves std::cout in fixed notation; print errors explicitly.
  std::ostringstream line;
  line << std::scientific << std::setprecision(3) << (ok ? "ok   " : "FAIL ")
       << name << " rows " << rows << " in " << in << " out " << out
       << ": worst " << worst << " (limit " << relative << ")\n";
  std::cout << line.str();
  if (!ok)
    ++failures;
}

void TestShape(std::size_t in, std::size_t out,
               const std::vector<std::size_t>& row_counts, std::mt19937& rng,
               hipStream_t stream) {
  const Q8Weights weights = MakeWeights(out, in, rng);
  const auto blocks = Upload(weights.blocks);
  vc::DeviceBuffer<std::uint8_t> q8_tiles(kernels::Q8TilesBytes(out, in));
  vc::Check(hipMemset(q8_tiles.get(), 0, q8_tiles.size()), "memset");
  kernels::Q8_0ToTiles(blocks.get(), out, in, q8_tiles.get(), stream);
  // The dequantized weights, as bf16 rows and bf16 tiles.
  vc::DeviceBuffer<std::uint16_t> dequantized(out * in);
  kernels::Q8TilesToBf16(q8_tiles.get(), out, in, dequantized.get(), stream);
  vc::DeviceBuffer<std::uint16_t> bf16_tiles(kernels::Bf16TilesBytes(out, in) /
                                             sizeof(std::uint16_t));
  vc::Check(hipMemset(bf16_tiles.get(), 0, kernels::Bf16TilesBytes(out, in)),
            "memset");
  kernels::Bf16ToTiles(dequantized.get(), out, in, bf16_tiles.get(), stream);
  vc::DeviceBuffer<std::uint16_t> unpacked(out * in);
  kernels::Bf16TilesToRows(bf16_tiles.get(), out, in, unpacked.get(), stream);
  std::vector<std::uint16_t> host_bf16(out * in);
  std::vector<std::uint16_t> host_unpacked(out * in);
  vc::Check(hipMemcpy(host_bf16.data(), dequantized.get(),
                      host_bf16.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost),
            "download");
  vc::Check(hipMemcpy(host_unpacked.data(), unpacked.get(),
                      host_unpacked.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost),
            "download");
  std::vector<double> bf16_values(out * in);
  bool dequant_ok = true;
  for (std::size_t i = 0; i < out * in; ++i) {
    bf16_values[i] = Bf16ToFloat(host_bf16[i]);
    if (host_bf16[i] != FloatToBf16(static_cast<float>(weights.values[i])))
      dequant_ok = false;
  }
  std::cout << (dequant_ok ? "ok   " : "FAIL ") << "dequantization in " << in
            << " out " << out << "\n";
  if (!dequant_ok)
    ++failures;
  const bool tiles_ok = host_unpacked == host_bf16;
  std::cout << (tiles_ok ? "ok   " : "FAIL ") << "bf16 tiles round trip in "
            << in << " out " << out << "\n";
  if (!tiles_ok)
    ++failures;

  vc::DeviceBuffer<float> partials(32 * 64 * out);
  vc::DeviceBuffer<unsigned> counters((out + 15) / 16);
  vc::Check(hipMemset(counters.get(), 0, counters.size() * sizeof(unsigned)),
            "memset");
  const kernels::SplitScratch scratch{partials.get(), partials.size(),
                                      counters.get(), counters.size()};
  for (const std::size_t rows : row_counts) {
    const std::vector<std::uint16_t> x = MakeActivations(rows, in, rng);
    const auto x_device = Upload(x);
    std::vector<float> before(rows * out);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float& value : before)
      value = normal(rng);
    vc::DeviceBuffer<float> y(rows * out);
    const auto run = [&](const std::string& name, bool accumulate,
                         const std::vector<double>& w, double limit,
                         const auto& product) {
      vc::Check(hipMemcpy(y.get(), before.data(), before.size() * sizeof(float),
                          hipMemcpyHostToDevice),
                "upload");
      if (!product(accumulate)) {
        std::cout << "FAIL " << name << " rows " << rows << " in " << in
                  << " out " << out << ": not launched\n";
        ++failures;
        return;
      }
      vc::Check(hipStreamSynchronize(stream), "sync");
      Compare(name + (accumulate ? " (accumulate)" : ""), Download(y, y.size()),
              x, w, accumulate ? before : std::vector<float>{}, rows, in, out,
              limit);
    };
    for (const bool accumulate : {false, true}) {
      for (const std::size_t waves :
           {std::size_t{1}, std::size_t{0}, std::size_t{4096}}) {
        const kernels::WmmaPlan plan = kernels::PlanWmma(rows, in, out, waves);
        const std::string suffix =
            " (" + std::to_string(plan.splits) + " splits)";
        run("q8 wmma" + suffix, accumulate, weights.values, 1e-5,
            [&](bool add) {
              return kernels::WmmaQ8(x_device.get(), q8_tiles.get(), y.get(),
                                     rows, in, out, add, plan, scratch, stream);
            });
        run("bf16 wmma" + suffix, accumulate, bf16_values, 1e-5, [&](bool add) {
          return kernels::WmmaBf16(x_device.get(), bf16_tiles.get(), y.get(),
                                   rows, in, out, add, plan, scratch, stream);
        });
      }
    }
  }
  std::vector<unsigned> left(counters.size());
  vc::Check(hipMemcpy(left.data(), counters.get(),
                      left.size() * sizeof(unsigned), hipMemcpyDeviceToHost),
            "download");
  for (const unsigned value : left) {
    if (value != 0) {
      std::cout << "FAIL split counters not reset\n";
      ++failures;
      break;
    }
  }
}

/// Bf16Gemm::Linear on Q8_0 and bf16 tiles: its split scratch, the chunks
/// of up to 128 rows and the expansion for hipBLASLt beyond.
void TestLinear(std::size_t in, std::size_t out, std::mt19937& rng,
                hipblasHandle_t blas, hipStream_t stream) {
  const Q8Weights weights = MakeWeights(out, in, rng);
  const auto blocks = Upload(weights.blocks);
  vc::DeviceBuffer<std::uint8_t> q8_tiles(kernels::Q8TilesBytes(out, in));
  vc::Check(hipMemset(q8_tiles.get(), 0, q8_tiles.size()), "memset");
  kernels::Q8_0ToTiles(blocks.get(), out, in, q8_tiles.get(), stream);
  vc::DeviceBuffer<std::uint16_t> dequantized(out * in);
  kernels::Q8TilesToBf16(q8_tiles.get(), out, in, dequantized.get(), stream);
  vc::DeviceBuffer<std::uint16_t> bf16_tiles(kernels::Bf16TilesBytes(out, in) /
                                             sizeof(std::uint16_t));
  vc::Check(hipMemset(bf16_tiles.get(), 0, kernels::Bf16TilesBytes(out, in)),
            "memset");
  kernels::Bf16ToTiles(dequantized.get(), out, in, bf16_tiles.get(), stream);
  std::vector<std::uint16_t> host_bf16(out * in);
  vc::Check(hipMemcpy(host_bf16.data(), dequantized.get(),
                      host_bf16.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost),
            "download");
  std::vector<double> bf16_values(out * in);
  for (std::size_t i = 0; i < out * in; ++i)
    bf16_values[i] = Bf16ToFloat(host_bf16[i]);
  vc::Bf16Gemm gemm(blas, stream);
  for (const std::size_t rows :
       {std::size_t{1}, std::size_t{22}, std::size_t{40}, std::size_t{130}}) {
    const std::vector<std::uint16_t> x = MakeActivations(rows, in, rng);
    const auto x_device = Upload(x);
    std::vector<float> before(rows * out);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (float& value : before)
      value = normal(rng);
    vc::DeviceBuffer<float> y(rows * out);
    for (const bool accumulate : {false, true}) {
      for (const bool q8 : {true, false}) {
        vc::Matrix matrix;
        matrix.format =
            q8 ? vc::WeightFormat::kQ8Tiles : vc::WeightFormat::kBf16Tiles;
        matrix.data = q8 ? static_cast<const void*>(q8_tiles.get())
                         : static_cast<const void*>(bf16_tiles.get());
        vc::Check(
            hipMemcpy(y.get(), before.data(), before.size() * sizeof(float),
                      hipMemcpyHostToDevice),
            "upload");
        gemm.Linear(x_device.get(), matrix, y.get(), rows, in, out, accumulate);
        vc::Check(hipStreamSynchronize(stream), "sync");
        // Beyond 128 rows Q8_0 is expanded to bf16(scale * q) for
        // hipBLASLt, whose reduction order differs.
        const bool expanded = rows > 128;
        Compare(std::string("linear ") + (q8 ? "q8" : "bf16") + " tiles" +
                    (accumulate ? " (accumulate)" : ""),
                Download(y, y.size()), x,
                q8 && !expanded ? weights.values : bf16_values,
                accumulate ? before : std::vector<float>{}, rows, in, out,
                expanded ? 1e-4 : 1e-5);
      }
    }
  }
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "no GPU; skipped\n";
    return kSkip;
  }
  hipStream_t stream = nullptr;
  vc::Check(hipStreamCreate(&stream), "stream");
  std::mt19937 rng(1234);
  const std::vector<std::size_t> all = {1, 2, 4, 5, 8, 11, 16, 22, 32};
  // VoxCPM2's shapes (feature and FSQ inputs, local stacks, LM, stop head),
  // odd row counts.
  TestShape(64, 1024, {4, 8}, rng, stream);
  TestShape(512, 1024, {1, 4}, rng, stream);
  TestShape(1024, 2560, all, rng, stream);
  TestShape(2048, 1024, {1, 5, 22}, rng, stream);
  TestShape(4096, 1024, {1, 5, 22}, rng, stream);
  TestShape(2048, 2048, {1, 5}, rng, stream);
  TestShape(6144, 2048, {1, 3}, rng, stream);
  TestShape(8192, 512, {1, 22}, rng, stream);
  TestShape(1024, 72, {3, 22}, rng, stream);  // rows not a multiple of 16
  TestShape(2048, 2, {1}, rng, stream);
  hipblasHandle_t blas = nullptr;
  vc::Check(hipblasCreate(&blas), "hipblasCreate");
  vc::Check(hipblasSetStream(blas, stream), "hipblasSetStream");
  TestLinear(1024, 2560, rng, blas, stream);
  TestLinear(4096, 1024, rng, blas, stream);
  vc::Check(hipblasDestroy(blas), "hipblasDestroy");
  vc::Check(hipStreamDestroy(stream), "stream");
  if (failures > 0) {
    std::cout << failures << " failures\n";
    return 1;
  }
  std::cout << "all products match\n";
  return 0;
}
