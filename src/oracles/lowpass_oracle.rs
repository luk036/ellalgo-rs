use std::f64::consts::PI;
// use ndarray::{stack, Axis, Array, Array1, Array2};
use crate::cutting_plane::{OracleFeas, OracleOptim};
use ndarray::{Array, Array1};

type Arr = Array1<f64>;
pub type Cut = (Arr, (f64, Option<f64>));

/// The `LowpassOracle` struct in Rust represents a lowpass filter with various configuration
/// parameters.
///
/// Properties:
///
/// * `more_alt`: The `more_alt` property is a boolean flag indicating whether there are more
///             alternative options available.
/// * `idx1`: The `idx1` property in the `LowpassOracle` struct is of type `i32`.
/// * `spectrum`: The `spectrum` property is a vector of type `Arr`.
/// * `nwpass`: The `nwpass` property in the `LowpassOracle` struct represents the number of points in
///             the passband of a lowpass filter.
/// * `nwstop`: The `nwstop` property in the `LowpassOracle` struct represents the number of points in
///             the stopband of a lowpass filter. It is used to determine the characteristics of the filter,
///             specifically the stopband width.
/// * `lp_sq`: The `lp_sq` property in the `LowpassOracle` struct appears to be a floating-point number
///            (f64). It likely represents a squared value used in low-pass filtering calculations or operations.
/// * `up_sq`: The `up_sq` property in the `LowpassOracle` struct appears to be a floating-point number
///            of type `f64`.
/// * `sp_sq`: The `sp_sq` property in the `LowpassOracle` struct represents a floating-point value of
///            type `f64`.
/// * `idx2`: The `idx2` property in the `LowpassOracle` struct appears to be a `i32` type. It is a
///           field that holds an unsigned integer value representing an index or position within the context of
///           the struct.
/// * `idx3`: The `idx3` property in the `LowpassOracle` struct represents an unsigned integer value.
/// * `fmax`: The `fmax` property in the `LowpassOracle` struct represents the maximum frequency value.
/// * `kmax`: The `kmax` property in the `LowpassOracle` struct represents the maximum value for a
///           specific type `i32`. It is used to store the maximum value for a certain index or count within the
///           context of the `LowpassOracle` struct.
pub struct LowpassOracle {
    pub more_alt: bool,
    pub idx1: i32,
    pub spectrum: Vec<Arr>,
    pub nwpass: i32,
    pub nwstop: i32,
    pub lp_sq: f64,
    pub up_sq: f64,
    pub sp_sq: f64,
    pub idx2: i32,
    pub idx3: i32,
    pub fmax: f64,
    pub kmax: i32,
}

impl LowpassOracle {
    /// The `new` function in Rust initializes a struct with specified parameters for spectral analysis.
    ///
    /// Arguments:
    ///
    /// * `ndim`: `ndim` represents the number of dimensions for the filter design.
    /// * `wpass`: The `wpass` parameter represents the passband edge frequency in the provided function.
    /// * `wstop`: The `wstop` parameter represents the stopband edge frequency in the given function.
    /// * `lp_sq`: The `lp_sq` parameter in the code represents the lower passband squared value. It is
    ///            used in the initialization of the struct and is a floating-point number (`f64`) passed as an
    ///            argument to the `new` function.
    /// * `up_sq`: The `up_sq` parameter in the function represents the upper bound squared value for
    ///            the filter design. It is used in the calculation and initialization of the struct fields in the
    ///            function.
    /// * `sp_sq`: The `sp_sq` parameter in the `new` function represents the square of the stopband
    ///            ripple level in the spectral domain. It is used in digital signal processing to define the
    ///            desired characteristics of a filter, specifically in this context for designing a filter with
    ///            given passband and stopband specifications.
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the struct that it belongs to. The struct
    /// contains several fields such as `more_alt`, `idx1`, `spectrum`, `nwpass`, `nwstop`, `lp_sq`,
    /// `up_sq`, `sp_sq`, `idx2`, `idx3`, `fmax`, and `kmax`. The function initializes these fields with
    /// the
    pub fn new(ndim: usize, wpass: f64, wstop: f64, lp_sq: f64, up_sq: f64, sp_sq: f64) -> Self {
        let mdim = 15 * ndim;
        let w: Array1<f64> = Array::linspace(0.0, std::f64::consts::PI, mdim);
        // let tmp: Array2<f64> = Array::from_shape_fn((mdim, ndim - 1), |(i, j)| 2.0 * (w[i] * (j + 1) as f64).cos());
        // let spectrum: Array2<f64> = stack![Axis(1), Array::ones(mdim).insert_axis(Axis(1)), tmp];

        let mut spectrum = vec![Arr::zeros(ndim); mdim];
        for i in 0..mdim {
            spectrum[i][0] = 1.0;
            for j in 1..ndim {
                spectrum[i][j] = 2.0 * (w[i] * j as f64).cos();
            }
        }
        // spectrum.iter_mut().for_each(|row| row.insert(0, 1.0));

        let nwpass = (wpass * (mdim - 1) as f64).floor() as i32 + 1;
        let nwstop = (wstop * (mdim - 1) as f64).floor() as i32 + 1;

        Self {
            more_alt: true,
            idx1: -1,
            spectrum,
            nwpass,
            nwstop,
            lp_sq,
            up_sq,
            sp_sq,
            idx2: nwpass - 1,
            idx3: nwstop - 1,
            fmax: f64::NEG_INFINITY,
            kmax: -1,
        }
    }
}

impl OracleFeas<Arr> for LowpassOracle {
    type CutChoice = (f64, Option<f64>); // parallel cut

    /// The `assess_feas` function in Rust assesses the feasibility of a given array `x` based on
    /// certain conditions and returns a corresponding `Cut` option.
    ///
    /// Arguments:
    ///
    /// * `x`: The `x` parameter in the `assess_feas` function is an array (`Arr`) that is passed by
    ///        reference (`&`). It is used to perform calculations and comparisons with the elements of the
    ///        `spectrum` array in the function.
    ///
    /// Returns:
    ///
    /// The function `assess_feas` returns an `Option` containing a tuple of type `Cut`. The `Cut` tuple
    /// consists of two elements: a vector of coefficients (`Arr`) and a tuple of two optional values.
    /// The first optional value represents the violation amount if the constraint is violated, and the
    /// second optional value represents the amount to reach feasibility if the constraint is
    /// infeasible.
    fn assess_feas(&mut self, x: &Arr) -> Option<Cut> {
        self.more_alt = true;

        let mdim = self.spectrum.len();
        let ndim = self.spectrum[0].len();
        for _ in 0..self.nwpass {
            self.idx1 += 1;
            if self.idx1 == self.nwpass {
                self.idx1 = 0;
            }
            let col_k = &self.spectrum[self.idx1 as usize];
            // let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let v = col_k.dot(x);
            if v > self.up_sq {
                let f = (v - self.up_sq, Some(v - self.lp_sq));
                return Some((col_k.clone(), f));
            }
            if v < self.lp_sq {
                let f = (-v + self.lp_sq, Some(-v + self.up_sq));
                return Some((col_k.iter().map(|&a| -a).collect(), f));
            }
        }

        self.fmax = f64::NEG_INFINITY;
        self.kmax = -1;
        for _ in self.nwstop..mdim as i32 {
            self.idx3 += 1;
            if self.idx3 == mdim as i32 {
                self.idx3 = self.nwstop;
            }
            let col_k = &self.spectrum[self.idx3 as usize];
            // let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let v = col_k.dot(x);
            if v > self.sp_sq {
                return Some((col_k.clone(), (v - self.sp_sq, Some(v))));
            }
            if v < 0.0 {
                return Some((
                    col_k.iter().map(|&a| -a).collect(),
                    (-v, Some(-v + self.sp_sq)),
                ));
            }
            if v > self.fmax {
                self.fmax = v;
                self.kmax = self.idx3;
            }
        }

        for _ in self.nwpass..self.nwstop {
            self.idx2 += 1;
            if self.idx2 == self.nwstop {
                self.idx2 = self.nwpass;
            }
            let col_k = &self.spectrum[self.idx2 as usize];
            // let v = col_k.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let v = col_k.dot(x);
            if v < 0.0 {
                // single cut
                return Some((col_k.iter().map(|&a| -a).collect(), (-v, None)));
            }
        }

        self.more_alt = false;

        if x[0] < 0.0 {
            let mut grad = Arr::zeros(ndim);
            grad[0] = -1.0;
            return Some((grad, (-x[0], None)));
        }

        None
    }
}

impl OracleOptim<Arr> for LowpassOracle {
    type CutChoice = (f64, Option<f64>); // parallel cut

    /// The function assess_optim takes in parameters x and sp_sq, updates the value of sp_sq, assesses
    /// feasibility of x, and returns a tuple containing a cut and a boolean value.
    ///
    /// Arguments:
    ///
    /// * `x`: The `x` parameter is of type `Arr`, which is likely an array or a slice of some kind. It
    ///         is passed by reference to the `assess_optim` function.
    /// * `sp_sq`: The `sp_sq` parameter in the `assess_optim` function is a mutable reference to a
    ///            `f64` value. This parameter is updated within the function and its value is used to determine
    ///            the return values of the function.
    fn assess_optim(&mut self, x: &Arr, sp_sq: &mut f64) -> (Cut, bool) {
        self.sp_sq = *sp_sq;

        if let Some(cut) = self.assess_feas(x) {
            return (cut, false);
        }

        let cut = (
            self.spectrum[self.kmax as usize].clone(),
            (0.0, Some(self.fmax)),
        );
        *sp_sq = self.fmax;
        (cut, true)
    }
}

/// The function `create_lowpass_case` in Rust calculates parameters for a lowpass filter based on given
/// values.
///
/// Arguments:
///
/// * `ndim`: The `ndim` parameter represents the number of dimensions for the lowpass filter. It is
///           used to create a `LowpassOracle` struct with specific parameters for the lowpass filter.
///
/// Returns:
///
/// A `LowpassOracle` struct is being returned with parameters `ndim`, `0.12`, `0.20`, `lp_sq`, `up_sq`,
/// and `sp_sq`.
pub fn create_lowpass_case(ndim: usize) -> LowpassOracle {
    let delta0_wpass = 0.025;
    let delta0_wstop = 0.125;
    let delta1 = 20.0 * (delta0_wpass * PI).log10();
    let delta2 = 20.0 * (delta0_wstop * PI).log10();

    let low_pass = 10.0f64.powf(-delta1 / 20.0);
    let up_pass = 10.0f64.powf(delta1 / 20.0);
    let stop_pass = 10.0f64.powf(delta2 / 20.0);

    let lp_sq = low_pass * low_pass;
    let up_sq = up_pass * up_pass;
    let sp_sq = stop_pass * stop_pass;

    LowpassOracle::new(ndim, 0.12, 0.20, lp_sq, up_sq, sp_sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    // use super::{ProfitOracle, ProfitOracleQ, ProfitRbOracle};
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use crate::ell::Ell;

    fn run_lowpass() -> (bool, usize) {
        let ndim = 32;
        let r0 = Arr::zeros(ndim);
        let mut ellip = Ell::new_with_scalar(40.0, r0);
        // ellip.helper.use_parallel_cut = use_parallel_cut;
        let mut omega = create_lowpass_case(ndim);
        let mut sp_sq = omega.sp_sq;
        let options = Options {
            max_iters: 50000,
            tolerance: 1e-14,
        };
        let (h, num_iters) = cutting_plane_optim(&mut omega, &mut ellip, &mut sp_sq, &options);
        (h.is_some(), num_iters)
    }

    #[test]
    fn test_lowpass() {
        let (_feasible, _num_iters) = run_lowpass();
        // assert!(feasible);
        // assert!(num_iters >= 23000);
        // assert!(num_iters <= 24000);
    }

    #[test]
    fn test_lowpass_oracle() {
        let mut oracle = create_lowpass_case(32);
        let x = Arr::zeros(32);
        let res = oracle.assess_feas(&x);
        assert!(res.is_some());
    }
}
