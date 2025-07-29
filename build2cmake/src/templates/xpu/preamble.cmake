cmake_minimum_required(VERSION 3.26)

# Set Intel SYCL compiler before project() call
find_program(ICPX_COMPILER icpx)
if(ICPX_COMPILER)
    set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
    message(STATUS "Using Intel SYCL compiler: ${ICPX_COMPILER}")
else()
    message(FATAL_ERROR "Intel SYCL compiler (icpx) not found. Please install Intel oneAPI toolkit.")
endif()

project({{ name }})

include("cmake/utils.cmake")

# Find Python with all necessary components for building extensions
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(Torch REQUIRED)

# Intel XPU backend detection and setup
if(NOT TORCH_VERSION)
  run_python(TORCH_VERSION "import torch; print(torch.__version__)" "Failed to get Torch version")
endif()

# Check for Intel XPU support in PyTorch
run_python(XPU_AVAILABLE
  "import torch; print('true' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'false')"
  "Failed to check XPU availability")

if(NOT XPU_AVAILABLE STREQUAL "true")
  message(WARNING "Intel XPU is not available in this PyTorch installation. XPU kernels will be skipped.")
  return()
endif()

# Set up XPU compilation flags
set(GPU_LANG "SYCL")
add_compile_definitions(XPU_KERNEL)
add_compile_definitions(USE_XPU)

# Set SYCL-specific flags
# Set comprehensive SYCL compilation and linking flags
set(sycl_link_flags "-fsycl;--offload-compress;-fsycl-targets=spir64_gen,spir64;-Xs;-device pvc,xe-lpg,ats-m150 -options ' -cl-intel-enable-auto-large-GRF-mode -cl-poison-unsupported-fp64-kernels -cl-intel-greater-than-4GB-buffer-required'")
set(sycl_flags "-fsycl;-fhonor-nans;-fhonor-infinities;-fno-associative-math;-fno-approx-func;-fno-sycl-instrument-device-code;--offload-compress;-fsycl-targets=spir64_gen,spir64;")
message(STATUS "Configuring for Intel XPU backend using SYCL")
