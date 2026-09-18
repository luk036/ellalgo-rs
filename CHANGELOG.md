# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Quickcheck integration for property-based testing
- Logging feature support with optional std dependency
- Property-based tests for core algorithms (100+ iterations per property)
- Custom Arbitrary implementations for test data generation

### Changed
- Improved test coverage with property-based testing approach
- Enhanced documentation with testing examples

### Fixed
- Resolved clippy warnings in lowpass_oracle.rs
- Fixed attribute placement for clippy lint suppression
- `bsearch` now stops when the bracket reaches floating-point resolution. The
  `tau < options.tolerance` test compared a scale-dependent width against an
  absolute constant, so once the bracket underflowed at its own magnitude the
  test was unreachable and the loop spun to `max_iters` without refining
  anything.

### Testing
- Added 5 property-based tests covering edge cases and invariants
- All 32 unit tests passing
- Integration with standard Rust testing workflow
- `test_bsearch_stops_at_float_resolution`: fails with 2000 iterations without
  the stall guard
