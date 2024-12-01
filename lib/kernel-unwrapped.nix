{
  kernelName,
  kernelSources,
  cudaCapabilities,
  src,

  lib,
  #stdenv,
  cudaPackages,
  cmake,
  ninja,

  # Remove, only here while we don't have a shim yet.
  torch
}:

let
  #ptx = lists.map (x: "${x}+PTX") cudaCapabilities;
  #capabilities = cudaCapabilities ++ ptx;

  dropDot = builtins.replaceStrings ["."] [""];
  stdenv = cudaPackages.backendStdenv;
in stdenv.mkDerivation {
  inherit src;

  name = "${kernelName}-unwrapped";

  nativeBuildInputs = [ cmake ninja cudaPackages.cuda_nvcc ];

  buildInputs = [
    torch
    torch.cxxdev
  ] ++ (with cudaPackages; [
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
    (lib.cmakeFeature "CMAKE_CUDA_ARCHITECTURES" (dropDot (lib.concatStringsSep ";" cudaCapabilities)))
  ];
}
