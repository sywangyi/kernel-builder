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

  srcFilter =
    src: name: type:
    type == "directory" || lib.any (suffix: lib.hasSuffix suffix name) src;

  # Build a single kernel.
  buildKernel =
    {
      name,
      path,
      buildConfig,
      pkgs,
    }:
    let
      src = builtins.path {
        inherit path;
        name = "${name}-src";
        filter = srcFilter buildConfig.src;
      };
    in
    pkgs.callPackage ./kernel {
      inherit src;
      inherit (pkgs.python3.pkgs) torch;
      kernelName = name;
      cudaCapabilities = buildConfig.capabilities;
      kernelSources = buildConfig.src;
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
    {
      path,
      pkgs,
      stripRPath ? false,
    }:
    let
      buildConfig = readBuildConfig path;
      extConfig = buildConfig.torch;
      src = builtins.path {
        inherit path;
        name = "${extConfig.name}-src";
        filter = srcFilter (extConfig.src ++ extConfig.pysrc);
      };
    in
    pkgs.callPackage ./torch-extension {
      inherit src stripRPath;
      inherit (pkgs.python3.pkgs) torch;
      extensionName = extConfig.name;
      extensionSources = extConfig.src;
      pySources = extConfig.pysrc;
      kernels = buildKernels path pkgs;
    };

  # Build a distributable Torch extension. In contrast to
  # `buildTorchExtension`, this flavor has the rpath stripped, making it
  # portable across Linux distributions. It also uses the build version
  # as the top-level directory.
  buildDistTorchExtension =
    path: pkgs:
    buildTorchExtension {
      inherit path pkgs;
      stripRPath = true;
    };

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

  buildTorchExtensionBundle =
    path:
    let
      # We just need to get any nixpkgs for use by the path join.
      pkgs = builtins.head pkgsForBuildConfigs;
      extensions = buildDistTorchExtensions path;
      namePaths = lib.mapAttrs (name: pkg: toString pkg) extensions;
    in
    import ./join-paths {
      inherit pkgs namePaths;
      name = "torch-ext-bundle";
    };
}
