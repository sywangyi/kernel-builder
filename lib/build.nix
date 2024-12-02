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
    torch:
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

    buildNixTorchExtensions = let
      # TODO: add other Torch and CUDA versions.
      torchVersions = with pkgs.python3Packages; [torch_2_4 ];
      flattenVersion = version: ''torch${lib.replaceStrings ["."] [""] (lib.versions.pad 2 version)}'';
      extensionForTorch = path: torch: { name = flattenVersion torch.version; value = buildTorchExtension path torch; };
    in
      path: builtins.listToAttrs (lib.map (extensionForTorch path) torchVersions);

    # TODO: rewrite rpaths.
    buildTorchExtensions = let
      stripRPath = drv: pkgs.runCommand "without-rpath" {} ''
        mkdir -p $out/lib
        find ${drv}/lib -name '*.so' -exec cp --no-preserve=mode {} $out/lib \;

        find $out/lib -name '*.so' \
          -exec patchelf --set-rpath '/opt/hostedtoolcache/Python/3.11.9/x64/lib' {} \;
      '';
    in
      path: lib.mapAttrs (v: drv: stripRPath drv) (buildNixTorchExtensions path);
}
