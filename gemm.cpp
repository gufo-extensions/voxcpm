// The plan cache follows Gufo's Qwen3-ASR hipBLASLt wrapper (MIT, see
// licenses/gufo.txt); algorithms are timed instead of taken by rank.
#include "gemm.hpp"

#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>

#include "device.hpp"
#include "kernels/skinny_gemm.hpp"
#include "kernels/wmma_products.hpp"

namespace gufo::models::voxcpm {
namespace {

struct Plan {
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatrixLayout_t a_layout = nullptr;
  hipblasLtMatrixLayout_t b_layout = nullptr;
  hipblasLtMatrixLayout_t c_layout = nullptr;
  hipblasLtMatmulAlgo_t algorithm{};
  std::size_t workspace_bytes = 0;
  bool usable = false;

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan() {
    if (c_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(c_layout);
    if (b_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(b_layout);
    if (a_layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(a_layout);
    if (operation != nullptr)
      (void)hipblasLtMatmulDescDestroy(operation);
  }
};

/// The fastest timed algorithm of a representative shape.
struct Choice {
  hipblasLtMatmulAlgo_t algorithm{};
  bool usable = false;
};

// Rows up to which tiles stay on the matrix cores (in chunks of 32).
constexpr std::size_t kChunkedRows = 128;
// Workspace for algorithms that split the reduction.
constexpr std::size_t kWorkspaceBytes = 32U << 20U;
constexpr int kCandidates = 32;
constexpr int kRepeats = 3;

hipblasOperation_t Operation(bool transpose) {
  return transpose ? HIPBLAS_OP_T : HIPBLAS_OP_N;
}

bool Layout(hipblasLtMatrixLayout_t* layout, hipDataType type, std::size_t rows,
            std::size_t columns, std::size_t ld, std::size_t batch,
            std::int64_t stride) {
  if (hipblasLtMatrixLayoutCreate(layout, type, rows, columns,
                                  static_cast<std::int64_t>(ld)) !=
      HIPBLAS_STATUS_SUCCESS)
    return false;
  if (batch <= 1)
    return true;
  const auto count = static_cast<std::int32_t>(batch);
  return hipblasLtMatrixLayoutSetAttribute(
             *layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count,
             sizeof(count)) == HIPBLAS_STATUS_SUCCESS &&
         hipblasLtMatrixLayoutSetAttribute(
             *layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
             sizeof(stride)) == HIPBLAS_STATUS_SUCCESS;
}

/// Elements a strided-batched column-major operand spans.
std::size_t Extent(std::size_t ld, std::size_t columns, std::int64_t stride,
                   std::size_t batch) {
  return static_cast<std::size_t>(stride) * (batch - 1) + ld * columns;
}

}  // namespace

struct Bf16Gemm::State {
  hipblasHandle_t fallback = nullptr;
  hipStream_t stream = nullptr;
  hipblasLtHandle_t handle = nullptr;
  DeviceBuffer<char> workspace;
  // Scratch operands for timing candidate algorithms.
  DeviceBuffer<std::uint16_t> tune_a, tune_b;
  DeviceBuffer<float> tune_c;
  std::map<MatmulShape, std::unique_ptr<Plan>> plans;  // exact shapes
  std::map<MatmulShape, Choice> tuned;                 // representatives
  // Split reductions of the matrix-core products; the counters are zero
  // between products.
  DeviceBuffer<float> partials;
  DeviceBuffer<unsigned> counters;
  // Tiles unpacked to bfloat16 rows for products of more than 128 rows.
  DeviceBuffer<std::uint16_t> unpacked;

  ~State() {
    plans.clear();
    if (handle != nullptr)
      (void)hipblasLtDestroy(handle);
  }

  bool Describe(Plan* plan, const MatmulShape& s) const {
    if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                  HIP_R_32F) != HIPBLAS_STATUS_SUCCESS)
      return false;
    const hipblasOperation_t a_operation = Operation(s.transpose_a);
    const hipblasOperation_t b_operation = Operation(s.transpose_b);
    if (hipblasLtMatmulDescSetAttribute(
            plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &a_operation,
            sizeof(a_operation)) != HIPBLAS_STATUS_SUCCESS ||
        (s.transpose_b &&
         hipblasLtMatmulDescSetAttribute(
             plan->operation, HIPBLASLT_MATMUL_DESC_TRANSB, &b_operation,
             sizeof(b_operation)) != HIPBLAS_STATUS_SUCCESS))
      return false;
    const std::size_t a_rows = s.transpose_a ? s.k : s.m;
    const std::size_t a_columns = s.transpose_a ? s.m : s.k;
    const std::size_t b_rows = s.transpose_b ? s.n : s.k;
    const std::size_t b_columns = s.transpose_b ? s.k : s.n;
    return Layout(&plan->a_layout, HIP_R_16BF, a_rows, a_columns, s.lda,
                  s.batch, s.stride_a) &&
           Layout(&plan->b_layout, HIP_R_16BF, b_rows, b_columns, s.ldb,
                  s.batch, s.stride_b) &&
           Layout(&plan->c_layout, HIP_R_32F, s.m, s.n, s.ldc, s.batch,
                  s.stride_c);
  }

  /// Times hipBLASLt's candidates for `s` on scratch operands; cached.
  Choice Tune(const MatmulShape& s) {
    if (const auto found = tuned.find(s); found != tuned.end())
      return found->second;
    Choice choice;
    Plan plan;
    if (!Describe(&plan, s)) {
      tuned.emplace(s, choice);
      return choice;
    }
    const std::size_t a_size =
        Extent(s.lda, s.transpose_a ? s.m : s.k, s.stride_a, s.batch);
    const std::size_t b_size =
        Extent(s.ldb, s.transpose_b ? s.k : s.n, s.stride_b, s.batch);
    const std::size_t c_size = Extent(s.ldc, s.n, s.stride_c, s.batch);
    tune_a.Reserve(a_size);
    tune_b.Reserve(b_size);
    tune_c.Reserve(c_size);
    Check(
        hipMemsetAsync(tune_a.get(), 0, a_size * sizeof(std::uint16_t), stream),
        "clear tuning operand");
    Check(
        hipMemsetAsync(tune_b.get(), 0, b_size * sizeof(std::uint16_t), stream),
        "clear tuning operand");
    hipblasLtMatmulPreference_t preference = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&preference) !=
        HIPBLAS_STATUS_SUCCESS) {
      tuned.emplace(s, choice);
      return choice;
    }
    const std::size_t capacity = workspace.size();
    (void)hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &capacity,
        sizeof(capacity));
    hipblasLtMatmulHeuristicResult_t results[kCandidates];
    int found = 0;
    const auto status = hipblasLtMatmulAlgoGetHeuristic(
        handle, plan.operation, plan.a_layout, plan.b_layout, plan.c_layout,
        plan.c_layout, preference, kCandidates, results, &found);
    (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (status != HIPBLAS_STATUS_SUCCESS || found <= 0) {
      tuned.emplace(s, choice);
      return choice;
    }
    hipEvent_t begin = nullptr, end = nullptr;
    Check(hipEventCreate(&begin), "hipEventCreate");
    Check(hipEventCreate(&end), "hipEventCreate");
    const float alpha = 1.0F;
    const float beta = s.accumulate ? 1.0F : 0.0F;
    float best = 0.0F;
    for (int index = 0; index < found; ++index) {
      const hipblasLtMatmulHeuristicResult_t& candidate = results[index];
      if (candidate.state != HIPBLAS_STATUS_SUCCESS ||
          candidate.workspaceSize > capacity)
        continue;
      const auto run = [&] {
        return hipblasLtMatmul(handle, plan.operation, &alpha, tune_a.get(),
                               plan.a_layout, tune_b.get(), plan.b_layout,
                               &beta, tune_c.get(), plan.c_layout, tune_c.get(),
                               plan.c_layout, &candidate.algo, workspace.get(),
                               candidate.workspaceSize, stream);
      };
      if (run() != HIPBLAS_STATUS_SUCCESS)
        continue;
      Check(hipEventRecord(begin, stream), "hipEventRecord");
      for (int repeat = 0; repeat < kRepeats; ++repeat)
        (void)run();
      Check(hipEventRecord(end, stream), "hipEventRecord");
      Check(hipEventSynchronize(end), "hipEventSynchronize");
      float milliseconds = 0.0F;
      Check(hipEventElapsedTime(&milliseconds, begin, end),
            "hipEventElapsedTime");
      if (!choice.usable || milliseconds < best) {
        best = milliseconds;
        choice.algorithm = candidate.algo;
        choice.usable = true;
      }
    }
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
    tuned.emplace(s, choice);
    return choice;
  }

  /// The exact shape's descriptors with its representative's algorithm.
  Plan* Resolve(const MatmulShape& shape, const MatmulShape& representative) {
    if (const auto found = plans.find(shape); found != plans.end())
      return found->second.get();
    auto owned = std::make_unique<Plan>();
    Plan* plan = owned.get();
    plans.emplace(shape, std::move(owned));
    if (handle == nullptr || !Describe(plan, shape))
      return plan;
    Choice choice = Tune(representative);
    if (!choice.usable)
      return plan;
    const float alpha = 1.0F;
    const float beta = shape.accumulate ? 1.0F : 0.0F;
    std::size_t bytes = 0;
    if (hipblaslt_ext::matmulIsAlgoSupported(
            handle, plan->operation, &alpha, plan->a_layout, plan->b_layout,
            &beta, plan->c_layout, plan->c_layout, choice.algorithm,
            bytes) != HIPBLAS_STATUS_SUCCESS ||
        bytes > workspace.size())
      return plan;
    plan->algorithm = choice.algorithm;
    plan->workspace_bytes = bytes;
    plan->usable = true;
    return plan;
  }
};

Bf16Gemm::Bf16Gemm(hipblasHandle_t fallback, hipStream_t stream)
    : state_(std::make_unique<State>()) {
  state_->fallback = fallback;
  state_->stream = stream;
  if (hipblasLtCreate(&state_->handle) != HIPBLAS_STATUS_SUCCESS) {
    state_->handle = nullptr;
    return;
  }
  state_->workspace = DeviceBuffer<char>(kWorkspaceBytes);
}

Bf16Gemm::~Bf16Gemm() = default;

void Bf16Gemm::Linear(const void* x, const Matrix& w, float* y,
                      std::size_t rows, std::size_t in, std::size_t out,
                      bool accumulate) {
  if (w.format == WeightFormat::kBf16) {
    LinearRows(x, w.data, y, rows, in, out, accumulate);
    return;
  }
  const bool q8 = w.format == WeightFormat::kQ8Tiles;
  hipStream_t stream = state_->stream;
  // Up to kChunkedRows rows, the matrix cores read the tiles once per 32
  // rows: as fast as expanding them for hipBLASLt (one 8-bit read against
  // an 8-bit read, a 16-bit write and a 16-bit read), and the Q8_0 values
  // stay exact.
  if (rows <= kChunkedRows) {
    const auto* activations = static_cast<const kernels::Bf16*>(x);
    bool launched = true;
    for (std::size_t first = 0; first < rows && launched; first += 32) {
      const std::size_t count = std::min<std::size_t>(32, rows - first);
      const kernels::WmmaPlan plan = kernels::PlanWmma(count, in, out);
      if (plan.splits > 1) {
        state_->partials.Reserve(plan.partial_floats);
        if (state_->counters.size() < plan.counters) {
          state_->counters.Reserve(plan.counters);
          Check(hipMemsetAsync(state_->counters.get(), 0,
                               state_->counters.size() * sizeof(unsigned),
                               stream),
                "clear split counters");
        }
      }
      const kernels::SplitScratch scratch{
          state_->partials.get(), state_->partials.size(),
          state_->counters.get(), state_->counters.size()};
      const kernels::Bf16* chunk_x = activations + first * in;
      float* chunk_y = y + first * out;
      launched =
          q8 ? kernels::WmmaQ8(
                   chunk_x, static_cast<const std::uint8_t*>(w.data), chunk_y,
                   count, in, out, accumulate, plan, scratch, stream)
             : kernels::WmmaBf16(
                   chunk_x, static_cast<const kernels::Bf16*>(w.data), chunk_y,
                   count, in, out, accumulate, plan, scratch, stream);
    }
    if (launched)
      return;
    throw GpuError("the matrix-core product does not suit its shape");
  }
  state_->unpacked.Reserve(out * in);
  if (q8)
    kernels::Q8TilesToBf16(static_cast<const std::uint8_t*>(w.data), out, in,
                           state_->unpacked.get(), stream);
  else
    kernels::Bf16TilesToRows(static_cast<const kernels::Bf16*>(w.data), out, in,
                             state_->unpacked.get(), stream);
  LinearRows(x, state_->unpacked.get(), y, rows, in, out, accumulate);
}

void Bf16Gemm::LinearRows(const void* x, const void* w, float* y,
                          std::size_t rows, std::size_t in, std::size_t out,
                          bool accumulate) {
  // A few rows: hipBLASLt's tiles waste most of their work, while the
  // product is bound by reading the weights once.
  if (kernels::SkinnyGemm(static_cast<const kernels::Bf16*>(x),
                          static_cast<const kernels::Bf16*>(w), y, rows, in,
                          out, accumulate, state_->stream))
    return;
  MatmulShape shape;
  shape.transpose_a = true;
  shape.m = out;
  shape.n = rows;
  shape.k = in;
  shape.lda = in;
  shape.ldb = in;
  shape.ldc = out;
  shape.accumulate = accumulate;
  MatmulShape representative = shape;
  representative.n = TuningBucket(rows);
  Matmul(shape, representative, w, x, y);
}

void Bf16Gemm::Matmul(const MatmulShape& shape,
                      const MatmulShape& representative, const void* a,
                      const void* b, float* c) {
  const float alpha = 1.0F;
  const float beta = shape.accumulate ? 1.0F : 0.0F;
  if (Plan* plan = state_->Resolve(shape, representative); plan->usable) {
    Check(hipblasLtMatmul(state_->handle, plan->operation, &alpha, a,
                          plan->a_layout, b, plan->b_layout, &beta, c,
                          plan->c_layout, c, plan->c_layout, &plan->algorithm,
                          state_->workspace.get(), plan->workspace_bytes,
                          state_->stream),
          "hipblasLtMatmul");
    return;
  }
  Check(hipblasGemmStridedBatchedEx(
            state_->fallback, Operation(shape.transpose_a),
            Operation(shape.transpose_b), static_cast<int>(shape.m),
            static_cast<int>(shape.n), static_cast<int>(shape.k), &alpha, a,
            HIP_R_16BF, static_cast<int>(shape.lda), shape.stride_a, b,
            HIP_R_16BF, static_cast<int>(shape.ldb), shape.stride_b, &beta, c,
            HIP_R_32F, static_cast<int>(shape.ldc), shape.stride_c,
            static_cast<int>(shape.batch), HIPBLAS_COMPUTE_32F,
            HIPBLAS_GEMM_DEFAULT),
        "hipblasGemmStridedBatchedEx");
}

}  // namespace gufo::models::voxcpm
