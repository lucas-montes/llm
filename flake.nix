{
  description = "A devShell example";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rust-bin-custom = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src"];
        };
      in {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          packages = with pkgs; [
            (with python312Packages; [
              venvShellHook
              torch
              tokenizers
              huggingface-hub
              safetensors
              sentencepiece
              jupyter
              ipython
              ipykernel
            ])
          ];

          buildInputs = [
            pkgs.pkg-config
            rust-bin-custom
          ];
        };
      }
    );
}
