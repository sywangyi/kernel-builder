define_gpu_extension_target(
  {{ ops_name }}
  DESTINATION {{ ops_name }}
  LANGUAGE ${GPU_LANG}
  SOURCES ${SRC}
  COMPILE_FLAGS ${SYCL_COMPILE_FLAGS}
  USE_SABI 3
  WITH_SOABI)

# Add XPU/SYCL specific linker flags
target_link_options({{ ops_name }} PRIVATE ${SYCL_LINK_FLAGS})
