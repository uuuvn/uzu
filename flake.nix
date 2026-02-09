{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
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
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            nixd
            uv
            cmake
            pkg-config
            openssl
            pkgs.llvmPackages.libclang
            wasmtime
            (rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
            cargo-nextest
            cargo-flamegraph
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          CPLUS_INCLUDE_PATH = let
            gccVersion = pkgs.stdenv.cc.cc.version;
          in
            builtins.concatStringsSep ":" [
              "${pkgs.stdenv.cc.cc}/include/c++/${gccVersion}"
              "${pkgs.stdenv.cc.cc}/include/c++/${gccVersion}/${pkgs.stdenv.hostPlatform.config}"
              "${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.libclang.version}/include"
              "${pkgs.glibc.dev}/include"
            ];
        };
      }
    );
}
