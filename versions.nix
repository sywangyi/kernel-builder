{ lib }:

rec {
  # Supported CUDA versions.
  cudaVersions = [
    "11.8"
    "12.1"
    "12.4"
  ];

  # All build configurations supported by Torch. We may want to split
  # this the CUDA versions by Torch version later, but the versions
  # are currently the same.
  buildConfigs = lib.cartesianProduct {
    cudaVersion = cudaVersions;
    torchVersion = [
      "2.4"
      "2.5"
    ];
    cxx11Abi = [
      true
      false
    ];
  };

}
