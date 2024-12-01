{ pkgs }:

rec {
  readToml = path: builtins.fromTOML (builtins.readFile path);

  readBuildConfig = path: readToml (path + "/build.toml");

  buildKernelUnwrapped = path: buildConfig: pkgs.callPackage ./kernel-unwrapped.nix {
    kernelName = buildConfig.kernel.name;
    cudaCapabilities = buildConfig.kernel.capabilities;
    kernelSources = buildConfig.kernel.sources;
    src = path;
    torch = pkgs.python3Packages.torch_2_4;
  };

  buildKernel =
    path:
    let
      buildConfig = readBuildConfig path;
    in
    buildKernelUnwrapped path buildConfig;
}
