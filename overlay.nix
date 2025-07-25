final: prev: {
  cmakeNvccThreadsHook = prev.callPackage ./pkgs/cmake-nvcc-threads-hook { };

  cmakeSyclHook = prev.callPackage ./pkgs/cmake-sycl-hook { };

  # Local packages

  build2cmake = prev.callPackage ./pkgs/build2cmake { };

  get-kernel-check = prev.callPackage ./pkgs/get-kernel-check { };

  kernel-abi-check = prev.callPackage ./pkgs/kernel-abi-check { };

  rewrite-nix-paths-macho = prev.callPackage ./pkgs/rewrite-nix-paths-macho { };

  stdenvGlibc_2_27 = prev.callPackage ./pkgs/stdenv-glibc-2_27 { };

  # Intel oneAPI toolkit
  inteloneapi-toolkit = prev.callPackage ./pkgs/inteloneapi-toolkit { };
}
