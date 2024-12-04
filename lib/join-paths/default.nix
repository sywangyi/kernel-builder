args@{
  pkgs,

  # Attribute set with names to paths.
  namePaths,

  preferLocalBuild ? true,
  allowSubstitutes ? false,
}:
let
  inherit (pkgs) lib;
  args_ = removeAttrs args [
    "pkgs"
    "namePaths"
  ];
  # Iterating over pairs in bash sucks, so let's generate
  # the commands in Nix instead.
  copyPath = path: pkg: ''
    mkdir -p ${placeholder "out"}/${path}
    cp -r ${pkg}/* ${placeholder "out"}/${path}
  '';
in
pkgs.runCommand "name" args_ (lib.concatStringsSep "\n" (lib.mapAttrsToList copyPath namePaths))
