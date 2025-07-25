{
  lib,
  stdenv,
  fetchurl,
  dpkg,
  autoPatchelfHook,
  makeWrapper,
  gcc,
  glibc,
  zlib,
  libxml2,
  ncurses,
  libuuid,
  xz,
  python3,
  version ? "2024.2.1",
}:

let
  # Intel oneAPI toolkit installer URLs for different components
  # These are the main components needed for SYCL/XPU development
  baseUrl = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/";
  
in stdenv.mkDerivation rec {
  pname = "inteloneapi-toolkit";
  inherit version;

  # For now, we'll create a minimal package that provides the necessary
  # environment and tools for SYCL development
  src = null;

  nativeBuildInputs = [
    makeWrapper
  ];

  buildInputs = [
    gcc
    glibc
    zlib
    libxml2
    ncurses
    libuuid
    xz
    python3
  ];

  # Since Intel oneAPI is proprietary and requires registration,
  # we'll create a placeholder that sets up the environment
  # Users will need to install oneAPI separately
  buildPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/include
    mkdir -p $out/lib
    mkdir -p $out/share/cmake
  '';

  installPhase = ''
    # Create wrapper scripts for SYCL tools
    cat > $out/bin/icpx << 'EOF'
#!/bin/bash
# Intel SYCL compiler wrapper
# This requires Intel oneAPI to be installed separately
if command -v /opt/intel/oneapi/compiler/latest/bin/icpx >/dev/null 2>&1; then
    exec /opt/intel/oneapi/compiler/latest/bin/icpx "$@"
elif command -v icpx >/dev/null 2>&1; then
    exec icpx "$@"
else
    echo "Error: Intel SYCL compiler (icpx) not found."
    echo "Please install Intel oneAPI toolkit from:"
    echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html"
    exit 1
fi
EOF
    chmod +x $out/bin/icpx

    cat > $out/bin/dpcpp << 'EOF'
#!/bin/bash
# Intel Data Parallel C++ compiler wrapper
if command -v /opt/intel/oneapi/compiler/latest/bin/dpcpp >/dev/null 2>&1; then
    exec /opt/intel/oneapi/compiler/latest/bin/dpcpp "$@"
elif command -v dpcpp >/dev/null 2>&1; then
    exec dpcpp "$@"
else
    echo "Error: Intel DPC++ compiler (dpcpp) not found."
    echo "Please install Intel oneAPI toolkit."
    exit 1
fi
EOF
    chmod +x $out/bin/dpcpp

    # Create a CMake config file for SYCL
    cat > $out/share/cmake/IntelSYCL.cmake << 'EOF'
# Intel SYCL CMake configuration
set(IntelSYCL_FOUND FALSE)

# Try to find Intel SYCL compiler
find_program(ICPX_COMPILER icpx PATHS
  /opt/intel/oneapi/compiler/latest/bin
  ENV PATH
)

if(ICPX_COMPILER)
  set(IntelSYCL_FOUND TRUE)
  set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
  
  # Set SYCL compilation flags
  set(SYCL_FLAGS "-fsycl")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")
  
  message(STATUS "Found Intel SYCL: ${ICPX_COMPILER}")
else()
  message(WARNING "Intel SYCL compiler not found. Please install Intel oneAPI toolkit.")
endif()
EOF

    # Create a version file
    echo "${version}" > $out/share/version
  '';

  passthru = {
    inherit version;
  };

  meta = with lib; {
    description = "Intel oneAPI Toolkit wrapper for SYCL development";
    longDescription = ''
      This package provides wrappers and environment setup for Intel oneAPI toolkit.
      The actual Intel oneAPI toolkit must be installed separately from Intel.
      
      This is a minimal package that enables SYCL development in the kernel-builder
      system by providing the necessary CMake configuration and compiler wrappers.
    '';
    homepage = "https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html";
    license = licenses.unfree;
    platforms = platforms.linux;
    maintainers = [ ];
  };
}
