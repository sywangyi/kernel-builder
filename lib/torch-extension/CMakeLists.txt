cmake_minimum_required(VERSION 3.13)
project(kernel LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development Development.SABIModule)

# Remove once we have the shim.
find_package(Torch REQUIRED)

if(NOT DEFINED EXTENSION_NAME)
  message(FATAL_ERROR "EXTENSION_NAME must be defined")
endif()


if(DEFINED EXTENSION_SOURCES)
  set(SOURCE_FILES ${EXTENSION_SOURCES})
else()
  message(FATAL_ERROR "EXTENSION_SOURCES must be defined")
endif()

if(NOT DEFINED KERNEL_LIBRARIES)
  message(FATAL_ERROR "EXTENSION_SOURCES must be defined")
endif()

set(C_EXTENSION "_${EXTENSION_NAME}")

Python_add_library(${C_EXTENSION} USE_SABI 3 WITH_SOABI MODULE "${SOURCE_FILES}")
target_compile_definitions(${C_EXTENSION} PRIVATE "-DTORCH_EXTENSION_NAME=${C_EXTENSION}")
target_link_libraries(${C_EXTENSION} PRIVATE CUDA::cudart CUDA::cuda_driver)

target_link_libraries(${C_EXTENSION} PRIVATE torch torch_python)
target_link_libraries(${C_EXTENSION} PRIVATE ${KERNEL_LIBRARIES})
set_target_properties(${C_EXTENSION} PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

install(TARGETS ${C_EXTENSION} LIBRARY COMPONENT ${C_EXTENSION} DESTINATION ${EXTENSION_NAME})

