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
        overlayForCudaVersion = cudaVersion: self: super: {
          cudaPackages =
            super."cudaPackages_${
              pkgs.lib.replaceStrings [ "." ] [ "_" ] (pkgs.lib.versions.pad 2 cudaVersion)
            }";
        };
        overlayForPythonVersion = pythonVersion: self: super: {
          python3 =
            super."python${pkgs.lib.replaceStrings [ "." ] [ "" ] (pkgs.lib.versions.pad 2 pythonVersion)}";
        };
        overlayForTorchVersion =
          { version, cxx11Abi }:
          self: super: {
            pythonPackagesExtensions = super.pythonPackagesExtensions ++ [
              (
                python-self: python-super: with python-self; {
                  torch =
                    python-super."torch_${
                      pkgs.lib.replaceStrings [ "." ] [ "_" ] (pkgs.lib.versions.pad 2 version)
                    }".override
                      { inherit cxx11Abi; };
                }
              )
            ];

          };
        pkgsForVersions =
          {
            cudaVersion,
            pythonVersion,
            torchVersion,
            cxx11Abi,
          }:
          import nixpkgs {
            inherit config system;
            overlays = [
              overlay
              (overlayForPythonVersion pythonVersion)
              (overlayForCudaVersion cudaVersion)
              (overlayForTorchVersion {
                inherit cxx11Abi;
                version = torchVersion;
              })
            ];
          };
        buildConfigs = pkgs.lib.cartesianProduct {
          cudaVersion = [
            "11.8"
            "12.1"
            "12.4"
          ];
          pythonVersion = [
            "3.9"
            "3.10"
            "3.11"
            "3.12"
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
      in
      rec {
        formatter = pkgs.nixfmt-rfc-style;
        lib = import lib/build.nix { inherit pkgs; };
        packages =
          rec {
            all = pkgs.symlinkJoin {
              name = "all";
              paths = pkgs.lib.attrsets.attrValues python3Packages;
            };

            python3Packages = with pkgs.python3.pkgs; {
              torch_2_4 = torch_2_4.override { cxx11Abi = false; };
              torch_2_4-cxx11Abi = torch_2_4.override { cxx11Abi = true; };
              torch_2_5 = torch_2_5.override { cxx11Abi = false; };
              torch_2_5-cxx11Abi = torch_2_5.override { cxx11Abi = true; };
            };

            cuda12_1 = pkgsForVersions {
              cudaVersion = "12.1";
              pythonVersion = "3.11";
              torchVersion = "2.4";
              cxx11Abi = false;
            };
            cuda11_8 = pkgsForVersions {
              cudaVersion = "11.8";
              pythonVersion = "3.10";
              torchVersion = "2.5";
              cxx11Abi = true;
            };

            python312 = pkgs.python312;
            python311 = pkgs.python311;
            python310 = pkgs.python310;
            python39 = pkgs.python39;
            python38 = pkgs.python38;

          }
          // (builtins.listToAttrs (
            map (
              buildConfig:
              let
                pkgs' = pkgsForVersions buildConfig;
              in
              {
                name = import ./lib/build-version.nix pkgs';
                value = pkgs'.python3.torch;
              }
            ) buildConfigs
          ));

      }
    )
    // {
      overlays.default = overlay;
    };
}
