{
  lib,
  makeSetupHook,
  inteloneapi-toolkit,
}:

makeSetupHook {
  name = "cmake-sycl-hook";
  propagatedBuildInputs = [ inteloneapi-toolkit ];
  substitutions = {
    inherit (inteloneapi-toolkit) version;
  };
  meta = with lib; {
    description = "CMake setup hook for Intel SYCL compilation";
    longDescription = ''
      This setup hook configures CMake to use Intel SYCL compiler (icpx)
      and sets appropriate SYCL compilation flags for XPU kernel development.
    '';
    platforms = platforms.linux;
  };
} ./cmake-sycl-hook.sh
