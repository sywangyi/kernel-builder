{
  buildConfi  xpuVersion = torch: "xpu${flattenVersion (lib.versions.majorMinor (pkgs.inteloneapi-toolkit.version or "2024.2"))}";
  
  # Check if any kernel uses XPU backend
  hasXpuKernel = lib.any (kernel: kernel.backend or null == "xpu") (lib.attrValues (buildConfig.kernel or {}));
  
  # Check if buildConfig explicitly enables XPU
  hasXpuConfig = buildConfig ? xpu && buildConfig.xpu == true; pkgs,
  torch,
  upstreamVariant,
}:
let
  inherit (pkgs) lib;
  inherit (import ./version-utils.nix { inherit lib; }) flattenVersion abiString;
  abi = torch: abiString torch.passthru.cxx11Abi;
  targetPlatform = pkgs.stdenv.targetPlatform.system;
  cudaVersion = torch: "cu${flattenVersion pkgs.cudaPackages.cudaMajorMinorVersion}";
  rocmVersion =
    torch: "rocm${flattenVersion (lib.versions.majorMinor pkgs.rocmPackages.rocm.version)}";
  xpuVersion = torch: "xpu${flattenVersion (lib.versions.majorMinor (pkgs.inteloneapi-toolkit.version or "2024.2"))}";
  
  # Check if any kernel uses XPU backend
  hasXpuKernel = lib.any (kernel: kernel.backend or null == "xpu") (lib.attrValues (buildConfig.kernel or {}));
  
  gpuVersion = torch: 
    if torch.cudaSupport then 
      cudaVersion torch
    else if torch.rocmSupport then 
      rocmVersion torch
    else if hasXpuKernel || hasXpuConfig then
      xpuVersion torch
    else if pkgs.stdenv.hostPlatform.isDarwin then
      "metal"
    else
      "cpu";
  torchVersion = torch: flattenVersion torch.version;
in
if pkgs.stdenv.hostPlatform.isDarwin then
  "torch${torchVersion torch}-metal-${targetPlatform}"
else
  "torch${torchVersion torch}-${abi torch}-${gpuVersion torch}-${targetPlatform}"
