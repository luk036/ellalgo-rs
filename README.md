# 🏉 ellalgo-rs

[![Crates.io](https://img.shields.io/crates/v/ellalgo-rs.svg)](https://crates.io/crates/ellalgo-rs)
[![Docs.rs](https://docs.rs/ellalgo-rs/badge.svg)](https://docs.rs/ellalgo-rs)
[![CI](https://github.com/luk036/ellalgo-rs/workflows/CI/badge.svg)](https://github.com/luk036/ellalgo-rs/actions)
[![codecov](https://codecov.io/gh/luk036/ellalgo-rs/branch/master/graph/badge.svg?token=KZnX3rl1gV)](https://codecov.io/gh/luk036/ellalgo-rs)

<p align="center">
  <img src="./ellipsoid-method.svg"/>
</p>

The Ellipsoid Method as a linear programming algorithm was first introduced by L. G. Khachiyan in 1979. It is a polynomial-time algorithm that uses ellipsoids to iteratively reduce the feasible region of a linear program until an optimal solution is found. The method works by starting with an initial ellipsoid that contains the feasible region, and then successively shrinking the ellipsoid until it contains the optimal solution. The algorithm is guaranteed to converge to an optimal solution in a finite number of steps.

The method has a wide range of practical applications in operations research. It can be used to solve linear programming problems, as well as more general convex optimization problems. The method has been applied to a variety of fields, including economics, engineering, and computer science. Some specific applications of the Ellipsoid Method include portfolio optimization, network flow problems, and the design of control systems. The method has also been used to solve problems in combinatorial optimization, such as the traveling salesman problem.

## What is Parallel Cut?
In the context of the Ellipsoid Method, a parallel cut refers to a pair of linear constraints of the form aTx <= b and -aTx <= -b, where a is a vector of coefficients and b is a scalar constant. These constraints are said to be parallel because they have the same normal vector a, but opposite signs. When a parallel cut is encountered during the Ellipsoid Method, both constraints can be used simultaneously to generate a new ellipsoid. This can improve the convergence rate of the method, especially for problems with many parallel constraints.

## Installation

### Cargo

* Install the rust toolchain in order to have cargo installed by following
  [this](https://www.rust-lang.org/tools/install) guide.
* run `cargo install ellalgo-rs`

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md).
