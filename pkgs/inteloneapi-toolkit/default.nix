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
  wget,
  gnupg,
  version ? "2024.2.1",
  autoInstall ? false,  # Set to true to attempt automatic installation
  components ? [ "dpcpp-cpp" "runtime" ],  # oneAPI components to install
}:

let
  # Intel oneAPI component downloads
  componentUrls = {
    "dpcpp-cpp" = {
      url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/89283f8a-c667-47b0-b7e1-c4573e37bd3e/l_dpcpp-cpp-compiler_p_${version}.79_offline.sh";
      sha256 = "1234567890abcdef"; # You'll need to get the actual hash
    };
    "runtime" = {
      url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/runtime_${version}_offline.sh";
      sha256 = "fedcba0987654321"; # You'll need to get the actual hash
    };
  };
  
  # Installation script for automatic setup
  installScript = ''
    #!/bin/bash
    set -e
    
    echo "Setting up Intel oneAPI automatic installation..."
    
    # Create temporary directory for installation
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    # Download and install using Intel's APT repository (Ubuntu/Debian)
    if command -v apt-get >/dev/null 2>&1; then
      echo "Using APT repository installation..."
      
      # Add Intel's GPG key
      wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > $TEMP_DIR/intel.gpg
      
      # Add repository
      echo "deb [signed-by=$TEMP_DIR/intel.gpg] https://apt.repos.intel.com/oneapi all main" > $TEMP_DIR/intel-oneapi.list
      
      # Install packages
      export DEBIAN_FRONTEND=noninteractive
      apt-get update
      apt-get install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-runtime-dpcpp-cpp
      
    # Download and install using YUM/DNF (Red Hat/Fedora)
    elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
      echo "Using YUM/DNF repository installation..."
      
      # Add Intel's YUM repository
      tee > /etc/yum.repos.d/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
      
      # Install packages
      if command -v dnf >/dev/null 2>&1; then
        dnf install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-runtime-dpcpp-cpp
      else
        yum install -y intel-oneapi-compiler-dpcpp-cpp intel-oneapi-runtime-dpcpp-cpp
      fi
      
    else
      echo "Package manager not supported for automatic installation."
      exit 1
    fi
  '';
  
in stdenv.mkDerivation rec {
  pname = "inteloneapi-toolkit";
  inherit version;

  # For automatic installation, fetch the installer
  src = if autoInstall && (builtins.hasAttr "dpcpp-cpp" componentUrls) then 
    fetchurl componentUrls.dpcpp-cpp
  else 
    null;

  nativeBuildInputs = [
    makeWrapper
  ] ++ lib.optionals autoInstall [
    wget
    gnupg
    autoPatchelfHook
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
  # Users will need to install oneAPI separately or enable autoInstall
  buildPhase = if autoInstall then ''
    echo "Attempting automatic Intel oneAPI installation..."
    
    # Create installation directory
    mkdir -p $out/opt/intel/oneapi
    
    # For automatic installation using Intel's installer
    if [ -n "$src" ]; then
      echo "Running Intel oneAPI installer..."
      chmod +x $src
      $src -a -s --eula accept --install-dir $out/opt/intel/oneapi
    fi
    
    # Create standard directories
    mkdir -p $out/bin
    mkdir -p $out/include  
    mkdir -p $out/lib
    mkdir -p $out/share/cmake
  '' else ''
    mkdir -p $out/bin
    mkdir -p $out/include
    mkdir -p $out/lib
    mkdir -p $out/share/cmake
  '';

  installPhase = ''
    # Enhanced wrapper scripts that work with both auto-installed and system oneAPI
    cat > $out/bin/icpx << 'EOF'
#!/bin/bash
# Intel SYCL compiler wrapper

# Check for auto-installed version first
if [ -x "@out@/opt/intel/oneapi/compiler/latest/bin/icpx" ]; then
    exec @out@/opt/intel/oneapi/compiler/latest/bin/icpx "$@"
# Check standard system locations
elif command -v /opt/intel/oneapi/compiler/latest/bin/icpx >/dev/null 2>&1; then
    exec /opt/intel/oneapi/compiler/latest/bin/icpx "$@"
elif command -v icpx >/dev/null 2>&1; then
    exec icpx "$@"
else
    echo "Error: Intel SYCL compiler (icpx) not found."
    echo "Please install Intel oneAPI toolkit from:"
    echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html"
    echo "Or rebuild this package with autoInstall = true"
    exit 1
fi
EOF
    chmod +x $out/bin/icpx

    cat > $out/bin/dpcpp << 'EOF'
#!/bin/bash
# Intel Data Parallel C++ compiler wrapper
if [ -x "@out@/opt/intel/oneapi/compiler/latest/bin/dpcpp" ]; then
    exec @out@/opt/intel/oneapi/compiler/latest/bin/dpcpp "$@"
elif command -v /opt/intel/oneapi/compiler/latest/bin/dpcpp >/dev/null 2>&1; then
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

    # Enhanced CMake config file for SYCL
    cat > $out/share/cmake/IntelSYCL.cmake << 'EOF'
# Intel SYCL CMake configuration
set(IntelSYCL_FOUND FALSE)

# Search paths for Intel SYCL compiler
set(INTEL_SEARCH_PATHS
  @out@/opt/intel/oneapi/compiler/latest/bin
  /opt/intel/oneapi/compiler/latest/bin
  $ENV{ONEAPI_ROOT}/compiler/latest/bin
)

find_program(ICPX_COMPILER 
  NAMES icpx
  PATHS ${INTEL_SEARCH_PATHS}
  NO_DEFAULT_PATH
)

if(NOT ICPX_COMPILER)
  find_program(ICPX_COMPILER icpx)
endif()

if(ICPX_COMPILER)
  set(IntelSYCL_FOUND TRUE)
  set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
  
  # Get oneAPI root from compiler path
  get_filename_component(ONEAPI_BIN_DIR ${ICPX_COMPILER} DIRECTORY)
  get_filename_component(ONEAPI_ROOT ${ONEAPI_BIN_DIR}/../../.. ABSOLUTE)
  
  # Set SYCL compilation and linking flags
  set(SYCL_COMPILE_FLAGS "-fsycl -fno-sycl-instrument-device-code")
  set(SYCL_LINK_FLAGS "-fsycl")
  
  # Add oneAPI include and library paths
  if(EXISTS ${ONEAPI_ROOT}/compiler/latest/include/sycl)
    include_directories(${ONEAPI_ROOT}/compiler/latest/include/sycl)
  endif()
  if(EXISTS ${ONEAPI_ROOT}/compiler/latest/lib)
    link_directories(${ONEAPI_ROOT}/compiler/latest/lib)
  endif()
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SYCL_LINK_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SYCL_LINK_FLAGS}")
  
  message(STATUS "Found Intel SYCL: ${ICPX_COMPILER}")
  message(STATUS "oneAPI root: ${ONEAPI_ROOT}")
else()
  message(WARNING "Intel SYCL compiler not found. Please install Intel oneAPI toolkit.")
endif()
EOF

    # Create environment setup script
    cat > $out/bin/setup-oneapi << 'EOF'
#!/bin/bash
# Setup Intel oneAPI environment

ONEAPI_ROOTS=(
  "@out@/opt/intel/oneapi"
  "/opt/intel/oneapi"
  "$HOME/intel/oneapi"
  "$ONEAPI_ROOT"
)

for root in "''${ONEAPI_ROOTS[@]}"; do
  if [ -n "$root" ] && [ -f "$root/setvars.sh" ]; then
    echo "Setting up oneAPI environment from: $root"
    source "$root/setvars.sh" --force
    return 0
  fi
done

echo "Warning: oneAPI environment setup not found."
echo "Available installation methods:"
echo "1. Install via Intel's installer: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html"
echo "2. Install via package manager:"
echo "   Ubuntu/Debian: apt install intel-oneapi-compiler-dpcpp-cpp"
echo "   RHEL/Fedora: dnf install intel-oneapi-compiler-dpcpp-cpp"
echo "3. Rebuild this package with autoInstall = true"
EOF
    chmod +x $out/bin/setup-oneapi

    # Create a version file
    echo "${version}" > $out/share/version
    
    # Substitute @out@ placeholders
    substituteInPlace $out/bin/icpx --replace "@out@" "$out"
    substituteInPlace $out/bin/dpcpp --replace "@out@" "$out"
    substituteInPlace $out/bin/setup-oneapi --replace "@out@" "$out"
    substituteInPlace $out/share/cmake/IntelSYCL.cmake --replace "@out@" "$out"
  '';

  # Fix dynamic linking for auto-installed version
  postFixup = lib.optionalString autoInstall ''
    if [ -d "$out/opt/intel/oneapi" ]; then
      # Fix paths and permissions
      find $out/opt/intel/oneapi -type f -executable | while read -r file; do
        if file "$file" | grep -q "ELF"; then
          echo "Patching: $file"
          autoPatchelf "$file" || true
        fi
      done
    fi
  '';

  passthru = {
    inherit version autoInstall components;
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
