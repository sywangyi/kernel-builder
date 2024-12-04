{
  kernelName,
  kernelSources,
  cudaCapabilities,
  src,

  lib,
  cudaPackages,
  cmake,
  ninja,

  # Remove, only here while we don't have a shim yet.
  torch,
}:

let
  dropDot = builtins.replaceStrings [ "." ] [ "" ];
  stdenv = cudaPackages.backendStdenv;
in
stdenv.mkDerivation {
  name = kernelName;

  inherit src;

  # Copy generic build files into the source tree.
  postPatch = ''
    cp ${./CMakeLists.txt} CMakeLists.txt
  '';

  nativeBuildInputs = [
    cmake
    ninja
    cudaPackages.cuda_nvcc
  ];

  buildInputs =
    [
      torch
      torch.cxxdev
    ]
    ++ (with cudaPackages; [
      cuda_cudart

      # Make dependent on build configuration dependencies once
      # the Torch dependency is gone.
      cuda_cccl
      libcublas
      libcusolver
      libcusparse
    ]);

  env = {
    CUDAToolkit_ROOT = "${lib.getDev cudaPackages.cuda_nvcc}";
    # Remove after removing torch dependency.
    TORCH_CUDA_ARCH_LIST = lib.concatStringsSep ";" cudaCapabilities;
  };

  cmakeFlags = [
    (lib.cmakeFeature "KERNEL_NAME" kernelName)
    (lib.cmakeFeature "KERNEL_SOURCES" (lib.concatStringsSep ";" kernelSources))
    (lib.cmakeFeature "CMAKE_CUDA_ARCHITECTURES" (dropDot (lib.concatStringsSep ";" cudaCapabilities)))
  ];
}
