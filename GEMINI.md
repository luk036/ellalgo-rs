# GEMINI.md

## Project Overview

This project, `ellalgo-rs`, is a Rust implementation of the Ellipsoid Method for solving linear programming and convex optimization problems. The library is designed to be used as a crate in other Rust projects. It includes the core Ellipsoid Method algorithm, as well as variations like using parallel cuts to improve convergence. The project also contains several examples of how to use the library.

The key technologies used are:
- **Language:** Rust
- **Core Dependencies:**
    - `ndarray`: For numerical linear algebra operations.
- **Development Tools:**
    - `cargo`: For building, testing, and dependency management.
    - `rustfmt`: For code formatting.
    - `clippy`: for linting.

The project is structured as a standard Rust library crate with all source code located in the `src` directory. It includes a `lib.rs` file as the main library entry point, and several modules for different components of the Ellipsoid Method. The `oracles` module provides different optimization problems to be solved.

## Building and Running

The project uses `cargo` for all build, run, and test operations.

### Key Commands:

- **Run all tests:**
  ```shell
  cargo test --all-features --workspace
  ```

- **Build and run in release mode:**
  ```shell
  cargo build --release && cargo run --release
  ```

- **Check for formatting issues:**
  ```shell
  cargo fmt --all --check
  ```

- **Format the code:**
  ```shell
  cargo fmt --all
  ```

- **Run Clippy for linting:**
  ```shell
  cargo clippy --all-targets
  ```

- **Generate documentation:**
  ```shell
  cargo doc --no-deps --document-private-items --all-features --workspace --examples
  ```

## Development Conventions

The `CONTRIBUTING.md` file outlines the process for contributing to the project.

- **Issues:** Check for existing issues before creating a new one.
- **Pull Requests:** One pull request per change.
- **Changelog:** Update `CHANGELOG.md` for any user-facing changes, following the "Keep a Changelog" format.
- **Code Style:** The project uses `rustfmt` to maintain a consistent code style. All code should be formatted before submission.
- **Linting:** `clippy` is used to catch common mistakes and improve code quality.
