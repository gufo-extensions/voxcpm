#ifndef GUFO_MODELS_VOXCPM_DEVICE_HPP_
#define GUFO_MODELS_VOXCPM_DEVICE_HPP_

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

// HIP helpers shared by the generator and the AudioVAE.
namespace gufo::models::voxcpm {

class GpuError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

inline void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw GpuError(std::string(what) + ": " + hipGetErrorString(status));
}

inline void Check(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw GpuError(std::string(what) + ": hipBLAS status " +
                   std::to_string(static_cast<int>(status)));
}

/// A device allocation of `count` elements of T.
template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { Resize(count); }
  ~DeviceBuffer() { Release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    Release();
    data_ = std::exchange(other.data_, nullptr);
    count_ = std::exchange(other.count_, 0);
    return *this;
  }

  /// Reallocates when `count` exceeds the capacity; contents are lost.
  void Reserve(std::size_t count) {
    if (count > count_)
      Resize(count);
  }

  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::size_t size() const { return count_; }

private:
  void Resize(std::size_t count) {
    Release();
    Check(hipMalloc(reinterpret_cast<void**>(&data_),
                    std::max<std::size_t>(count, 1) * sizeof(T)),
          "hipMalloc");
    count_ = count;
  }
  void Release() noexcept {
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    count_ = 0;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace gufo::models::voxcpm

#endif  // GUFO_MODELS_VOXCPM_DEVICE_HPP_
