{ pkgs }:

rec {
  inherit (pkgs) lib;

  readToml = path: builtins.fromTOML (builtins.readFile path);

  readBuildConfig = path: readToml (path + "/build.toml");

  buildKernelUnwrapped = { name, path, buildConfig }: pkgs.callPackage ./kernel {
    kernelName = name;
    cudaCapabilities = buildConfig.capabilities;
    kernelSources = buildConfig.src;
    src = path;
    torch = pkgs.python3Packages.torch_2_4;
  };

  buildKernels =
    path:
    let
      buildConfig = readBuildConfig path;
      kernels = lib.mapAttrs (name: buildConfig: buildKernelUnwrapped {inherit name path buildConfig;}) buildConfig.kernel;
    in
      kernels;

  buildTorchExtension =
    path:
    let
      buildConfig = readBuildConfig path;
      extConfig = buildConfig.extension;
    in
    pkgs.callPackage ./torch-extension {
      extensionName = extConfig.name;
      extensionSources = extConfig.src;
      kernels = buildKernels path;
      src = path;
      torch = pkgs.python3Packages.torch_2_4;
    };
 
}
