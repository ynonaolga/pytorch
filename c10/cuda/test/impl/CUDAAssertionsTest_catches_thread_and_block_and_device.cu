#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using ::testing::HasSubstr;

/**
 * TEST: Triggering device side assertion on only 1 thread from <<<1024,128>>>
 * grid. kernel used is unique, it take 2 parameters to tell which particular
 * block and thread it should assert, all the other theads of the kernel will be
 * basically no-op.
 */
void cuda_device_assertions_catches_thread_and_block_and_device() {
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_DSA_KERNEL_LAUNCH(
      cuda_device_assertions_fail_on_thread_block_kernel,
      1024, /* Blocks */
      128, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      29, /* bad thread */
      937 /* bad block */
  );

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(
        err_str, HasSubstr("Thread ID that failed assertion = [29,0,0]"));
    ASSERT_THAT(
        err_str, HasSubstr("Block ID that failed assertion = [937,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_device_assertions_fail_on_thread_block_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream.id())));
  }
}

int main() {
  cuda_device_assertions_catches_thread_and_block_and_device();

  return 0;
}
