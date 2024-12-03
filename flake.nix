{
  description = "Kernel builder";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable-small";
    flake-compat.url = "github:edolstra/flake-compat";
  };

  outputs =
    {
      self,
      flake-compat,
      flake-utils,
      nixpkgs,
    }:
    let
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };

      overlay = import ./overlay.nix;
    in
    flake-utils.lib.eachSystem [ flake-utils.lib.system.x86_64-linux ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit config system;
          overlays = [ overlay ];
        };
        buildVersion = import ./lib/build-version.nix;
        overlayForCudaVersion = cudaVersion: self: super: {
          cudaPackages =
            super."cudaPackages_${
              self.lib.replaceStrings [ "." ] [ "_" ] (self.lib.versions.pad 2 cudaVersion)
            }";
        };
        overlayForTorchVersion =
          { version, cxx11Abi }:
          self: super: {
            pythonPackagesExtensions = super.pythonPackagesExtensions ++ [
              (
                python-self: python-super: with python-self; {
                  torch =
                    python-super."torch_${
                      self.lib.replaceStrings [ "." ] [ "_" ] (self.lib.versions.pad 2 version)
                    }".override
                      { inherit cxx11Abi; };
                }
              )
            ];

          };

        # Construct the nixpkgs package set for the given versions.
        pkgsForVersions =
          {
            cudaVersion,
            torchVersion,
            cxx11Abi,
          }:
          import nixpkgs {
            inherit config system;
            overlays = [
              overlay
              (overlayForCudaVersion cudaVersion)
              (overlayForTorchVersion {
                inherit cxx11Abi;
                version = torchVersion;
              })
            ];
          };

        # All build configurations supported by Torch.
        buildConfigs = pkgs.lib.cartesianProduct {
          cudaVersion = [
            "11.8"
            "12.1"
            "12.4"
          ];
          torchVersion = [
            "2.4"
            "2.5"
          ];
          cxx11Abi = [
            true
            false
          ];
        };

        pkgsForBuildConfigs = map pkgsForVersions buildConfigs;
      in
      rec {
        formatter = pkgs.nixfmt-rfc-style;
        lib = import lib/build.nix {
          inherit (pkgs) lib;
          inherit pkgsForBuildConfigs;
        };
        packages = rec {
          # This package set is exposed so that we can prebuild the Torch versions.
          torch = builtins.listToAttrs (
            map (pkgs': {
              name = buildVersion pkgs';
              value = pkgs'.python3.pkgs.torch;
            }) pkgsForBuildConfigs
          );
        };
      }
    );
}
