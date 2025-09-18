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
        rust-bin-custom = pkgs.rust-bin.stable."1.88.0".default.override {
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
              numpy
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

            # Debugging tools
            pkgs.lldb # LLDB debugger (works with CodeLLDB extension)
            pkgs.gdb # GDB debugger (alternative)
            pkgs.valgrind # Memory debugging

            # Development tools
            pkgs.strace # System call tracing
            pkgs.ltrace # Library call tracing
            pkgs.perf-tools # Performance profiling
            pkgs.flamegraph # Flamegraph generation for benchmarks

            # Optional: more debugging utilities
            pkgs.rr # Record and replay debugger
            pkgs.heaptrack # Heap memory profiler
          ];
        };
      }
    );
}
