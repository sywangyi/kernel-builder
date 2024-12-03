{ pkgs }:

let
  inherit (pkgs) lib;
  flattenVersion = version: lib.replaceStrings [ "." ] [ "" ] (lib.versions.pad 2 version);
  abi = torch: if torch.passthru.cxx11Abi then "cxx11" else "cxx98";
  targetPlatform = pkgs.stdenv.targetPlatform.system;
  cudaVersion = torch: flattenVersion torch.cudaPackages.cudaMajorMinorVersion;
  pythonVersion = torch: flattenVersion torch.pythonModule.version;
  torchVersion = torch: flattenVersion torch.version;
  torchBuildVersion =
    torch:
    "torch${torchVersion torch}-${abi torch}-cu${cudaVersion torch}-py${pythonVersion torch}-${targetPlatform}";

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
    }:
    pkgs.callPackage ./kernel {
      kernelName = name;
      cudaCapabilities = buildConfig.capabilities;
      kernelSources = buildConfig.src;
      src = path;
      torch = pkgs.python3Packages.torch_2_4;
    };

  # Build all kernels defined in build.toml.
  buildKernels =
    path:
    let
      buildConfig = readBuildConfig path;
      kernels = lib.mapAttrs (
        name: buildConfig: buildKernel { inherit name path buildConfig; }
      ) buildConfig.kernel;
    in
    kernels;

  # Build a single Torch extension.
  buildTorchExtension =
    path: torch:
    let
      buildConfig = readBuildConfig path;
      extConfig = buildConfig.extension;
    in
    pkgs.callPackage ./torch-extension {
      inherit torch;
      extensionName = extConfig.name;
      extensionSources = extConfig.src;
      kernels = buildKernels path;
      src = path;
    };

  # Build a distributable Torch extension. In contrast to
  # `buildTorchExtension`, this flavor has the rpath stripped, making it
  # portable across Linux distributions. It also uses the build version
  # as the top-level directory.
  buildDistTorchExtension =
    path: torch:
    let
      pkg = buildTorchExtension path torch;
      buildVersion = torchBuildVersion torch;
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
      torchVersions = with pkgs.python3Packages; [ torch_2_4 ];
      extensionForTorch = path: torch: {
        name = torchBuildVersion torch;
        value = buildTorchExtension path torch;
      };
    in
    path: builtins.listToAttrs (lib.map (extensionForTorch path) torchVersions);

  # Build multiple Torch extensions.
  buildDistTorchExtensions =
    let
      torchVersions = with pkgs.python3Packages; [ torch_2_4 ];
      extensionForTorch = path: torch: {
        name = torchBuildVersion torch;
        value = buildDistTorchExtension path torch;
      };
    in
    path: builtins.listToAttrs (lib.map (extensionForTorch path) torchVersions);

}
