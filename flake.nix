{
  description = "cavity polaritons from electric field response";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    poetry2nix,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
      pkgs = nixpkgs.legacyPackages.${system};
      #inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication;
      inherit (poetry2nix.lib.mkPoetry2Nix {inherit pkgs;}) mkPoetryEnv;
      qedmftEnv = mkPoetryEnv {
        projectDir = ./.;
        extras = ["jup"];
        editablePackageSources = {
          # added PROJECT_DIR var in .envrc
          # https://github.com/nix-community/poetry2nix/issues/425
          #qedmft = if builtins.getEnv "PROJECT_DIR" == "" then ./src else builtins.getEnv "PROJECT_DIR";
          # still didn't work? what about hard coding it?
          qedmft = /home/john/git/qedmft/src;
          #qedmft = ./src; # UGH...
          # when in quotes package is editable, but need to import src.qedmft
          # when not in quotes can import qedmft, but it's not editable
          # appending dirs means I can only import individual files
          # even worse: with it in quotes it doesn't appear in jupyter at all wtf?
        };
      };
    in {
      devShells.default = qedmftEnv.env;

      #devShells.default = qedmftEnv;
      ##packages = {
      ##  qedmft = mkPoetryApplication { projectDir = ./.;
      ##                                 extras = [ "jup" ];};
      ##  default = self.packages.${system}.qedmft;
      ##};

      ##devShells.default = pkgs.mkShell {
      ##  inputsFrom = [ self.packages.${system}.qedmft ];
      ##  packages = [ pkgs.poetry ];
      ##};
    });
}
