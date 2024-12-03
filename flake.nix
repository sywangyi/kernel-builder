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
      in
      rec {
        formatter = pkgs.nixfmt-rfc-style;
        lib = import lib/build.nix { inherit pkgs; };
        packages = rec {
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
        };
      }
    )
    // {
      overlays.default = overlay;
    };
}
