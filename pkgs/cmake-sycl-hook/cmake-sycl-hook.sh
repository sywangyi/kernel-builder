#!/bin/bash

# CMake SYCL Hook
# This hook sets up the environment for Intel SYCL compilation

cmakeSyclHook() {
    echo "Setting up Intel SYCL environment for CMake..."
    
    # Set CMake variables for SYCL
    if [ -n "${cmakeFlags-}" ]; then
        cmakeFlags="$cmakeFlags -DCMAKE_CXX_COMPILER=icpx"
        cmakeFlags="$cmakeFlags -DCMAKE_CXX_FLAGS=-fsycl"
        cmakeFlags="$cmakeFlags -DIntelSYCL_FOUND=TRUE"
    else
        cmakeFlags="-DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS=-fsycl -DIntelSYCL_FOUND=TRUE"
    fi
    
    # Add SYCL include paths if available
    if [ -d "/opt/intel/oneapi/compiler/latest/include/sycl" ]; then
        cmakeFlags="$cmakeFlags -DSYCL_INCLUDE_DIR=/opt/intel/oneapi/compiler/latest/include"
    fi
    
    # Set environment variables for Intel oneAPI
    if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
        echo "Found Intel oneAPI environment script"
        # Note: We don't source it here as it might interfere with the build
        # Instead, we just note its presence
    fi
    
    export cmakeFlags
    echo "CMake SYCL flags: $cmakeFlags"
}

# Add the hook to be called during the configure phase
if [ -z "${dontUseCmakeSyclHook-}" ] && [ -z "${dontUseCmakeConfigureHook-}" ]; then
    preConfigureHooks+=(cmakeSyclHook)
fi
