pkgs:
let
  inherit (pkgs) lib;
  inherit (pkgs.python3.pkgs) torch;
  flattenVersion = version: lib.replaceStrings [ "." ] [ "" ] (lib.versions.pad 2 version);
  abi = torch: if torch.passthru.cxx11Abi then "cxx11" else "cxx98";
  targetPlatform = pkgs.stdenv.targetPlatform.system;
  cudaVersion = torch: flattenVersion torch.cudaPackages.cudaMajorMinorVersion;
  torchVersion = torch: flattenVersion torch.version;
in
"torch${torchVersion torch}-${abi torch}-cu${cudaVersion torch}-${targetPlatform}"
