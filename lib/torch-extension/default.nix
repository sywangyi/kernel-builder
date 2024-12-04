{
  extensionName,
  extensionSources,

  # Keys are kernel names, values derivations.
  kernels,

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
  name = "${extensionName}-torch-ext";

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
    ])
    ++ (lib.attrValues kernels);

  env = {
    CUDAToolkit_ROOT = "${lib.getDev cudaPackages.cuda_nvcc}";
    # Remove after removing torch dependency.
    #TORCH_CUDA_ARCH_LIST = lib.concatStringsSep ";" cudaCapabilities;
  };

  cmakeFlags =
    let
      kernelPath = name: drv: "${drv}/lib/lib${name}.a";
      kernelLibs = lib.mapAttrsToList kernelPath kernels;
    in
    [
      (lib.cmakeFeature "EXTENSION_NAME" extensionName)
      (lib.cmakeFeature "EXTENSION_SOURCES" (lib.concatStringsSep ";" extensionSources))
      (lib.cmakeFeature "KERNEL_LIBRARIES" (lib.concatStringsSep " " kernelLibs))
      #(lib.cmakeFeature "CMAKE_CUDA_ARCHITECTURES" (dropDot (lib.concatStringsSep ";" cudaCapabilities)))
    ];
}
