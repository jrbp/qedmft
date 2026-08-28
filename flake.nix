{
  description = "cavity polaritons from electric field response";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    ...
  }: let
    inherit (nixpkgs) lib;
    forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    project = pyproject-nix.lib.project.loadPyproject {
      projectRoot = ./.;
    };
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      default = let
        # there's gotta be a better way than this, right?
        python = pkgs.python3.override {
          self = python;
          packageOverrides = pyfinal: pyprev: {
            qedmft = pyfinal.mkPythonEditablePackage (
              project.renderers.mkPythonEditablePackage {
                inherit python;
                root = "$PROJECT_DIR";
              }
            );
          };
        };
        pythonEnv = python.withPackages (ps: [ps.qedmft]);
      in
        pkgs.mkShell {
          packages = [pythonEnv];
        };
      depsOnly = let
        python = pkgs.python3;
        pythonEnv = python.withPackages (project.renderers.withPackages {inherit python;});
      in
        pkgs.mkShell {
          packages = [pythonEnv];
        };
    });
    packages = forAllSystems (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in rec {
        default = qedmft;
        qedmft = python.pkgs.buildPythonPackage (
          project.renderers.buildPythonPackage {inherit python;}
        );
        pythonWithQedmft = pkgs.python3.withPackages (_: [self.packages.${system}.qedmft]);
      }
    );
  };
}
