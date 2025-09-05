[
  {
    torchVersion = "2.7";
    cudaVersion = "11.8";
    cxx11Abi = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.7";
    cudaVersion = "12.6";
    cxx11Abi = true;
    systems = [
      "x86_64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.7";
    cudaVersion = "12.8";
    cxx11Abi = true;
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.7";
    rocmVersion = "6.3.4";
    cxx11Abi = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.7";
    rocmVersion = "6.4.2";
    cxx11Abi = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = false;
  }
  {
    torchVersion = "2.7";
    xpuVersion = "2025.0.2";
    cxx11Abi = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.8";
    xpuVersion = "2025.1.3";
    cxx11Abi = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.7";
    cxx11Abi = true;
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }

  {
    torchVersion = "2.8";
    cudaVersion = "12.6";
    cxx11Abi = true;
    systems = [
      "x86_64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.8";
    cudaVersion = "12.8";
    cxx11Abi = true;
    systems = [
      "x86_64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.8";
    cudaVersion = "12.9";
    cxx11Abi = true;
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.8";
    rocmVersion = "6.3.4";
    cxx11Abi = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.8";
    rocmVersion = "6.4.2";
    cxx11Abi = true;
    systems = [ "x86_64-linux" ];
    bundleBuild = true;
  }
  {
    torchVersion = "2.8";
    cxx11Abi = true;
    metal = true;
    systems = [ "aarch64-darwin" ];
    bundleBuild = true;
  }

  # Non-standard versions; not included in bundle builds.
  {
    torchVersion = "2.7";
    cudaVersion = "12.9";
    cxx11Abi = true;
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  }
  {
    torchVersion = "2.8";
    cudaVersion = "12.4";
    cxx11Abi = true;
    systems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  }

]
