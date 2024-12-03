{
  lib,

  # List of packages sets, where each has a different CUDA/Torch
  # specialization.
  pkgsForBuildConfigs,
}:

let
  flattenVersion = version: lib.replaceStrings [ "." ] [ "" ] (lib.versions.pad 2 version);
  abi = torch: if torch.passthru.cxx11Abi then "cxx11" else "cxx98";
  torchBuildVersion = import ./build-version.nix;
in
rec {

  readToml = path: builtins.fromTOML (builtins.readFile path);

  readBuildConfig = path: readToml (path + "/build.toml");

  # Build a single kernel.
  buildKernel =
    {
      name,
      path,
      buildConfig,
      pkgs,
    }:
    pkgs.callPackage ./kernel {
      inherit (pkgs.python3.pkgs) torch;
      kernelName = name;
      cudaCapabilities = buildConfig.capabilities;
      kernelSources = buildConfig.src;
      src = path;
    };

  # Build all kernels defined in build.toml.
  buildKernels =
    path: pkgs:
    let
      buildConfig = readBuildConfig path;
      kernels = lib.mapAttrs (
        name: buildConfig:
        buildKernel {
          inherit
            name
            path
            buildConfig
            pkgs
            ;
        }
      ) buildConfig.kernel;
    in
    kernels;

  # Build a single Torch extension.
  buildTorchExtension =
    path: pkgs:
    let
      buildConfig = readBuildConfig path;
      extConfig = buildConfig.extension;
    in
    pkgs.callPackage ./torch-extension {
      inherit (pkgs.python3.pkgs) torch;
      extensionName = extConfig.name;
      extensionSources = extConfig.src;
      kernels = buildKernels path pkgs;
      src = path;
    };

  # Build a distributable Torch extension. In contrast to
  # `buildTorchExtension`, this flavor has the rpath stripped, making it
  # portable across Linux distributions. It also uses the build version
  # as the top-level directory.
  buildDistTorchExtension =
    path: pkgs:
    let
      pkg = buildTorchExtension path pkgs;
      buildVersion = torchBuildVersion pkgs;
    in
    pkgs.runCommand buildVersion { } ''
      mkdir -p $out/${buildVersion}
      find ${pkg}/lib -name '*.so' -exec cp --no-preserve=mode {} $out/${buildVersion} \;

      find $out/${buildVersion} -name '*.so' \
        -exec patchelf --set-rpath '/opt/hostedtoolcache/Python/3.11.9/x64/lib' {} \;
    '';

  # Build multiple Torch extensions.
  buildNixTorchExtensions =
    let
      torchVersions = map (pkgs: pkgs.python3.pkgs.torch) pkgsForBuildConfigs;
      extensionForTorch = path: pkgs: {
        name = torchBuildVersion pkgs;
        value = buildTorchExtension path pkgs;
      };
    in
    path: builtins.listToAttrs (lib.map (extensionForTorch path) pkgsForBuildConfigs);

  # Build multiple Torch extensions.
  buildDistTorchExtensions =
    let
      extensionForTorch = path: pkgs: {
        name = torchBuildVersion pkgs;
        value = buildDistTorchExtension path pkgs;
      };
    in
    path: builtins.listToAttrs (lib.map (extensionForTorch path) pkgsForBuildConfigs);

}
