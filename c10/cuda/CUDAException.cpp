#include <c10/cuda/CUDAException.h>

#include <c10/util/Backtrace.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#define CHECK_CUDA_API_CALL_WITHOUT_CHECKING_DEVICE_ASSERTS \
  c10_cuda_check_implementation(__FILE__, __FUNCTION__, __LINE__, false)

namespace c10 {
namespace cuda {

/// Check that kernels ran correctly by checking the message buffer. BLOCKING.
std::string c10_retrieve_device_side_assertion_info() {
#ifdef TORCH_USE_CUDA_DSA
  const auto& launch_registry = CUDAKernelLaunchRegistry::get_singleton_ref();
  if (!launch_registry.enabled) {
    return "Device-side assertion tracking was not enabled.";
  }

  // Hack that saves a lot of challenging sync logic.
  // The GPU increments the number of errors it's observed and the CPU can see
  // that happening immediately which means we can make it here before the GPU
  // is done writing information about those errors to memory.
  // A short pause gives it time to finish. Since something's gone wrong, this
  // pause shouldn't affect perf.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // The snapshot causes a brief block. That's okay because this function only
  // executes if something's gone wrong such that speed is no longer a priority.
  const auto launch_data = launch_registry.snapshot();
  const auto& assertion_data = launch_data.first;
  const auto& launch_infos = launch_data.second;

  std::stringstream oss;

  oss << "# of GPUs this process interacted with = "
      << launch_registry.gpus_interacted_with() << std::endl;

  // Loop over each device that could be managed by the process
  for (const auto device_num : c10::irange(assertion_data.size())) {
    const auto& assertion_data_for_device = assertion_data.at(device_num);

    // Did anything fail?
    const auto failures_found = std::min(
        assertion_data_for_device.assertion_count,
        C10_CUDA_DSA_ASSERTION_COUNT);
    if (failures_found == 0) {
      continue;
    }

    // Something failed, let's talk about that
    oss << failures_found
        << " CUDA device-side assertion failures were found on GPU #"
        << device_num << "!" << std::endl;
    if (assertion_data_for_device.assertion_count >
        C10_CUDA_DSA_ASSERTION_COUNT) {
      oss << "But at least " << assertion_data_for_device.assertion_count
          << " assertion failures occurred on the device" << std::endl;
      oss << "Adjust `C10_CUDA_DSA_ASSERTION_COUNT` if you need more assertion failure info"
          << std::endl;
    }

    for (const auto i : c10::irange(failures_found)) {
      const auto& self = assertion_data_for_device.assertions[i];
      const auto& launch_info = launch_infos[self.caller % launch_infos.size()];
      oss << "Assertion failure " << i << std::endl;
      oss << "  GPU assertion failure message = " << self.assertion_msg
          << std::endl;
      oss << "  File containing assertion = " << self.filename << ":"
          << self.line_number << std::endl;
      oss << "  Device function containing assertion = " << self.function_name
          << std::endl;
      oss << "  Thread ID that failed assertion = [" << self.thread_id[0] << ","
          << self.thread_id[1] << "," << self.thread_id[2] << "]" << std::endl;
      oss << "  Block ID that failed assertion = [" << self.block_id[0] << ","
          << self.block_id[1] << "," << self.block_id[2] << "]" << std::endl;
      if (launch_info.generation_number == self.caller) {
        oss << "  File containing kernel launch = "
            << launch_info.launch_filename << ":" << launch_info.launch_linenum
            << std::endl;
        oss << "  Function containing kernel launch = "
            << launch_info.launch_function << std::endl;
        oss << "  Name of kernel launched that led to failure = "
            << launch_info.kernel_name << std::endl;
        oss << "  Device that launched kernel = " << launch_info.device
            << std::endl;
        oss << "  Stream kernel was launched on = " << launch_info.stream
            << std::endl;
        oss << "  Backtrace of kernel launch site = ";
        if (launch_registry.gather_launch_stacktrace) {
          oss << "Launch stacktracing disabled." << std::endl;
        } else {
          oss << "\n" << launch_info.launch_stacktrace << std::endl;
        }
      } else {
        oss << "  CPU launch site info: Unavailable, the circular queue wrapped around. Increase `CUDAKernelLaunchRegistry::max_size`."
            << std::endl;
      }
    }
  }
  return oss.str();
#else
  return "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n";
#endif
}

void c10_cuda_check_implementation(
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions) {
  const auto cuda_error = cudaGetLastError();
  const auto cuda_kernel_failure = include_device_assertions
      ? c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed()
      : false;

  if (C10_LIKELY(cuda_error == cudaSuccess && !cuda_kernel_failure)) {
    return;
  }

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("CUDA API error: ");
  check_message.append(cudaGetErrorString(cuda_error));
  check_message.append(c10::cuda::get_cuda_check_suffix());
  check_message.append("\n");
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif

  TORCH_CHECK(false, check_message);
}

CUDAKernelLaunchRegistry::CUDAKernelLaunchRegistry()
    : do_all_devices_support_managed_memory(
          check_if_all_devices_support_managed_memory()),
      gather_launch_stacktrace(check_env_for_enable_launch_stacktracing()),
      enabled(check_env_for_dsa_enabled()) {
  for (C10_UNUSED const auto _ : c10::irange(get_device_count())) {
    uvm_assertions.emplace_back(nullptr, uvm_deleter);
  }

  kernel_launches.resize(max_kernel_launches);
}

bool CUDAKernelLaunchRegistry::env_flag_set(const char* env_var_name) {
  const char* const env_string = std::getenv(env_var_name);
  return (env_string == nullptr) ? false : std::strcmp(env_string, "0");
}

bool CUDAKernelLaunchRegistry::check_env_for_enable_launch_stacktracing()
    const {
  return env_flag_set("PYTORCH_CUDA_DSA_STACKTRACING");
}

bool CUDAKernelLaunchRegistry::check_env_for_dsa_enabled() const {
  return env_flag_set("PYTORCH_USE_CUDA_DSA");
}

uint32_t CUDAKernelLaunchRegistry::insert(
    const char* launch_filename,
    const char* launch_function,
    const uint32_t launch_linenum,
    const char* kernel_name,
    const int32_t stream_id) {
#ifdef TORCH_USE_CUDA_DSA
  if (!enabled) {
    return 0;
  }
  const std::lock_guard<std::mutex> lock(read_write_mutex);
  const auto my_gen_number = generation_number++;
  // TODO: It would probably be good to get a stack trace here so that
  // we can better indicate which launch caused the failure.
  kernel_launches[my_gen_number % max_kernel_launches] = {
      launch_filename,
      launch_function,
      launch_linenum,
      gather_launch_stacktrace ? c10::get_backtrace() : "",
      kernel_name,
      get_device_id(),
      stream_id,
      my_gen_number};
  return my_gen_number;
#else
  return 0;
#endif
}

std::pair<std::vector<DeviceAssertionsData>, std::vector<CUDAKernelLaunchInfo>>
CUDAKernelLaunchRegistry::snapshot() const {
  // This is likely to be the longest-lasting hold on the mutex, but
  // we only expect it to be called in cases where we're already failing
  // and speed is no longer important
  const std::lock_guard<std::mutex> lock(read_write_mutex);

  std::vector<DeviceAssertionsData> device_assertions_data;
  for (const auto& x : uvm_assertions) {
    if (x) {
      device_assertions_data.push_back(*x);
    } else {
      device_assertions_data.emplace_back();
    }
  }

  return std::make_pair(device_assertions_data, kernel_launches);
}

DeviceAssertionsData* CUDAKernelLaunchRegistry::
    get_uvm_assertions_ptr_for_current_device() {
#ifdef TORCH_USE_CUDA_DSA
  if (!enabled || !do_all_devices_support_managed_memory) {
    return nullptr;
  }

  const auto device_num = get_device_id();

  // If we've already set up this GPU with managed memory, return a pointer to
  // the managed memory. This is a lock-free quick-return path.
  if (uvm_assertions.at(device_num)) {
    return uvm_assertions.at(device_num).get();
  }

  // Need a lock here so there's not race-condition on creating the new device
  // assertions buffer
  const std::lock_guard<std::mutex> lock(gpu_alloc_mutex);

  // If we've already set up this GPU with managed memory, return a pointer to
  // the managed memory. This locked path ensures that the device memory is
  // allocated only once
  if (uvm_assertions.at(device_num)) {
    return uvm_assertions.at(device_num).get();
  }

  // Otherwise, set up the GPU to be able to use the device-side assertion
  // system
  DeviceAssertionsData* uvm_assertions_ptr = nullptr;

  C10_CUDA_ERROR_HANDLED(
      cudaMallocManaged(&uvm_assertions_ptr, sizeof(DeviceAssertionsData)));
  CHECK_CUDA_API_CALL_WITHOUT_CHECKING_DEVICE_ASSERTS;

  C10_CUDA_ERROR_HANDLED(cudaMemAdvise(
      uvm_assertions_ptr,
      sizeof(DeviceAssertionsData),
      cudaMemAdviseSetPreferredLocation,
      cudaCpuDeviceId));
  CHECK_CUDA_API_CALL_WITHOUT_CHECKING_DEVICE_ASSERTS;

  // GPU will establish direct mapping of data in CPU memory, no page faults
  // will be generated
  C10_CUDA_ERROR_HANDLED(cudaMemAdvise(
      uvm_assertions_ptr,
      sizeof(DeviceAssertionsData),
      cudaMemAdviseSetAccessedBy,
      cudaCpuDeviceId));
  CHECK_CUDA_API_CALL_WITHOUT_CHECKING_DEVICE_ASSERTS;

  // Initialize the memory from the CPU; otherwise, pages may have to be created
  // on demand. We think that UVM documentation indicates that first access may
  // not honor preferred location, which would be bad, if true, because we want
  // this memory on the host so we can access it post-assertion. Initializing
  // this on the CPU helps ensure that that's where the memory will live.
  *uvm_assertions_ptr = DeviceAssertionsData();

  // Ownership and lifetime management of `uvm_assertions_ptr` now passes to the
  // uvm_assertions unique_ptr vector
  uvm_assertions.at(device_num).reset(uvm_assertions_ptr);

  return uvm_assertions_ptr;
#else
  return nullptr;
#endif
}

// Class needs its own implementation of this function to prevent
// an infinite initialization loop for CUDAKernelLaunchRegistry
int CUDAKernelLaunchRegistry::get_device_count() {
  int device_count = -1;
  C10_CUDA_ERROR_HANDLED(cudaGetDeviceCount(&device_count));
  CHECK_CUDA_API_CALL_WITHOUT_CHECKING_DEVICE_ASSERTS;
  return device_count;
}

// Class needs its own implementation of this function to prevent
// an infinite initialization loop for CUDAKernelLaunchRegistry
int CUDAKernelLaunchRegistry::get_device_id() {
  int device = -1;
  C10_CUDA_ERROR_HANDLED(cudaGetDevice(&device));
  CHECK_CUDA_API_CALL_WITHOUT_CHECKING_DEVICE_ASSERTS;
  return device;
}

// Class needs its own implementation of this function to prevent
// an infinite initialization loop for CUDAKernelLaunchRegistry
int CUDAKernelLaunchRegistry::get_device_compute_capability(
    const int device_num) {
  int compute_capability = -1;
  C10_CUDA_ERROR_HANDLED(cudaDeviceGetAttribute(
      &compute_capability, cudaDevAttrComputeCapabilityMajor, device_num));
  CHECK_CUDA_API_CALL_WITHOUT_CHECKING_DEVICE_ASSERTS;
  return compute_capability;
}

bool CUDAKernelLaunchRegistry::check_if_all_devices_support_managed_memory() {
// It looks as though this'll work best on CUDA GPUs with Pascal
// architectures or newer, per
// https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
#ifdef TORCH_USE_CUDA_DSA
  for (const auto i : c10::irange(get_device_count())) {
    if (get_device_compute_capability(i) < 6) {
      return false;
    }
  }
  return true;
#else
  return false;
#endif
}

void CUDAKernelLaunchRegistry::uvm_deleter(
    DeviceAssertionsData* uvm_assertions_ptr) {
  // Ignore error in destructor
  if (uvm_assertions_ptr) {
    C10_CUDA_IGNORE_ERROR(cudaFree(uvm_assertions_ptr));
  }
}

CUDAKernelLaunchRegistry& CUDAKernelLaunchRegistry::get_singleton_ref() {
  static CUDAKernelLaunchRegistry launch_registry;
  return launch_registry;
}

int CUDAKernelLaunchRegistry::gpus_interacted_with() const {
  int gpu_count = 0;
  for (const auto& x : uvm_assertions) {
    if (x) {
      gpu_count++;
    }
  }

  return gpu_count;
}

bool CUDAKernelLaunchRegistry::has_failed() const {
  for (const auto& x : uvm_assertions) {
    if (x && x->assertion_count > 0) {
      return true;
    }
  }
  return false;
}

} // namespace cuda
} // namespace c10
