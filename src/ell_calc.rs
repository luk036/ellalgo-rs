// mod lib;
use crate::cutting_plane::CutStatus;

/// The `EllCalcCore` struct represents the parameters for calculating the new Ellipsoid Search Space.
///
/// Properties:
///
/// * `n_f`: The `n_f` property represents the number of variables in the ellipsoid search space.
/// * `n_plus_1`: The `n_plus_1` property represents the value of `n + 1`, where `n` is the dimension of
/// the search space. It is used in calculations related to the ellipsoid search space.
/// * `half_n`: The `half_n` property represents half of the dimension of the ellipsoid search space. It
/// is used in the calculation of the parameters for the ellipsoid search space.
/// * `cst1`: The `cst1` property is a constant used in the calculation of the parameters for the new
/// Ellipsoid Search Space. Its specific purpose and value are not provided in the code snippet.
/// * `cst2`: The `cst2` property is a constant used in the calculation of the parameters for the new
/// Ellipsoid Search Space. It is not specified what exactly it represents or how it is used in the
/// calculation.
/// * `cst3`: The `cst3` property is a constant value used in the calculation of the parameters for the
/// new Ellipsoid Search Space. It is not specified what exactly this constant represents or how it is
/// used in the calculations.
#[derive(Debug, Clone)]
pub struct EllCalcCore {
    pub n_f: f64,
    pub n_plus_1: f64,
    pub half_n: f64,
    pub inv_n: f64,
    cst1: f64,
    cst2: f64,
}

impl EllCalcCore {
    /// Constructs a new EllCalcCore instance, initializing its fields based on the provided dimension n_f.
    ///
    /// This is a constructor function for the EllCalcCore struct. It takes in the dimension n_f and uses
    /// it to calculate and initialize the other fields of EllCalcCore that depend on n_f.
    ///
    /// Being a constructor function, it is part of the public API of the EllCalcCore struct. The
    /// documentation follows the conventions and style of the other documentation in this crate.
    ///
    /// Arguments:
    ///
    /// * `n_f`: The parameter `n_f` represents the value of `n` in the calculations. It is a floating-point
    /// number.
    ///
    /// Returns:
    ///
    /// The `new` function returns an instance of the `EllCalcCore` struct.
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalcCore;
    ///
    /// let ell_calc_core = EllCalcCore::new(4.0);
    ///
    /// assert_approx_eq!(ell_calc_core.n_f, 4.0);
    /// assert_approx_eq!(ell_calc_core.half_n, 2.0);
    /// assert_approx_eq!(ell_calc_core.n_plus_1, 5.0);
    /// ```
    pub fn new(n_f: f64) -> EllCalcCore {
        let n_plus_1 = n_f + 1.0;
        let half_n = n_f / 2.0;
        let inv_n = 1.0 / n_f;
        let n_sq = n_f * n_f;
        let cst0 = 1.0 / (n_f + 1.0);
        let cst1 = n_sq / (n_sq - 1.0);
        let cst2 = 2.0 * cst0;

        EllCalcCore {
            n_f,
            n_plus_1,
            half_n,
            inv_n,
            cst1,
            cst2,
        }
    }

    #[doc = svgbobdoc::transform!(
    /// The function calculates the core values for updating an ellipsoid with either a parallel-cut or
    /// a deep-cut.
    ///
    /// Arguments:
    ///
    /// * `beta0`: The parameter `beta0` represents the semi-minor axis of the ellipsoid before the cut. It is
    /// a floating-point number.
    /// * `beta1`: The parameter `beta1` represents the length of the semi-minor axis of the ellipsoid.
    /// * `tsq`: tsq is a reference to a f64 value, which represents the square of the semi-major axis
    /// of the ellipsoid.
    ///
    /// ```svgbob
    ///      _.-'''''''-._
    ///    ,'     |       `.
    ///   /  |    |         \
    ///  .   |    |          .
    ///  |   |    |          |
    ///  |   |    |.         |
    ///  |   |    |          |
    ///  :\  |    |         /:
    ///  | `._    |      _.' |
    ///  |   |'-.......-'    |
    ///  |   |    |          |
    /// "-τ" "-β" "-β"      +τ
    ///        1    0
    /// 
    ///      β  + β                            
    ///       0    1                           
    ///  β = ───────                           
    ///         2                              
    ///                                        
    ///      1   ⎛ 2          ⎞        2       
    ///  h = ─ ⋅ ⎜τ  + β  ⋅ β ⎟ + n ⋅ β        
    ///      2   ⎝      0    1⎠                
    ///             _____________________      
    ///            ╱ 2                  2      
    ///  k = h + ╲╱ h  - (n + 1) ⋅ η ⋅ β       
    ///                                        
    ///        1     η                         
    ///  σ = ───── = ─                         
    ///      μ + 1   k                         
    ///                                        
    ///  1     η                               
    ///  ─ = ─────                             
    ///  μ   k - η                             
    ///                                        
    ///  ϱ = β ⋅ σ                            
    ///                                        
    ///       2    2   1   ⎛ 2              ⎞  
    ///  δ ⋅ τ  = τ  + ─ ⋅ ⎜β  ⋅ σ - β  ⋅ β ⎟  
    ///                μ   ⎝          0    1⎠   
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalcCore;
    ///
    /// let ell_calc_core = EllCalcCore::new(4.0);
    /// let (rho, sigma, delta) = ell_calc_core.calc_parallel_bias_cut_fast(1.0, 2.0, 4.0, 2.0, 12.0);
    /// assert_approx_eq!(rho, 1.2);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(delta, 0.8);
    /// ```
    )]
    pub fn calc_parallel_bias_cut_fast(
        &self,
        beta0: f64,
        beta1: f64,
        tsq: f64,
        b0b1: f64,
        eta: f64,
    ) -> (f64, f64, f64) {
        let bavg = (beta0 + beta1) * 0.5;
        let bavgsq = bavg * bavg;
        let h = (tsq + b0b1) * 0.5 + self.n_f * bavgsq;
        let k = h + (h * h - eta * self.n_plus_1 * bavgsq).sqrt();
        let inv_mu_plus_1 = eta / k;
        let inv_mu = eta / (k - eta);
        let rho = bavg * inv_mu_plus_1;
        let sigma = inv_mu_plus_1;
        let delta = (tsq + inv_mu * (bavgsq * inv_mu_plus_1 - b0b1)) / tsq;

        (rho, sigma, delta)
    }

    #[doc = svgbobdoc::transform!(
    /// The function calculates the core values for updating an ellipsoid with either a parallel-cut or
    /// a deep-cut.
    ///
    /// Arguments:
    ///
    /// * `beta0`: The parameter `beta0` represents the semi-minor axis of the ellipsoid before the cut. It is
    /// a floating-point number.
    /// * `beta1`: The parameter `beta1` represents the length of the semi-minor axis of the ellipsoid.
    /// * `tsq`: tsq is a reference to a f64 value, which represents the square of the semi-major axis
    /// of the ellipsoid.
    ///
    /// ```svgbob
    ///      _.-'''''''-._
    ///    ,'     |       `.
    ///   /  |    |         \
    ///  .   |    |          .
    ///  |   |    |          |
    ///  |   |    |.         |
    ///  |   |    |          |
    ///  :\  |    |         /:
    ///  | `._    |      _.' |
    ///  |   |'-.......-'    |
    ///  |   |    |          |
    /// "-τ" "-β" "-β"      +τ
    ///        1    0
    /// 
    ///       2                            
    ///  η = τ  + n ⋅ β  ⋅ β               
    ///                0    1                
    ///      β  + β                            
    ///       0    1                           
    ///  β = ───────                           
    ///         2                              
    ///                                        
    ///      1   ⎛ 2          ⎞        2       
    ///  h = ─ ⋅ ⎜τ  + β  ⋅ β ⎟ + n ⋅ β        
    ///      2   ⎝      0    1⎠                
    ///             _____________________      
    ///            ╱ 2                  2      
    ///  k = h + ╲╱ h  - (n + 1) ⋅ η ⋅ β       
    ///                                        
    ///        1     η                         
    ///  σ = ───── = ─                         
    ///      μ + 1   k                         
    ///                                        
    ///  1     η                               
    ///  ─ = ─────                             
    ///  μ   k - η                             
    ///                                        
    ///  ϱ = β ⋅ σ                            
    ///                                        
    ///       2    2   1   ⎛ 2              ⎞  
    ///  δ ⋅ τ  = τ  + ─ ⋅ ⎜β  ⋅ σ - β  ⋅ β ⎟  
    ///                μ   ⎝          0    1⎠   
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalcCore;
    ///
    /// let ell_calc_core = EllCalcCore::new(4.0);
    /// let (rho, sigma, delta) = ell_calc_core.calc_parallel_bias_cut(1.0, 2.0, 4.0);
    /// assert_approx_eq!(rho, 1.2);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(delta, 0.8);
    /// ```
    )]
    #[inline]
    pub fn calc_parallel_bias_cut(&self, beta0: f64, beta1: f64, tsq: f64) -> (f64, f64, f64) {
        let b0b1 = beta0 * beta1;
        let eta = tsq + self.n_f * b0b1;
        self.calc_parallel_bias_cut_fast(beta0, beta1, tsq, b0b1, eta)
    }

    #[doc = svgbobdoc::transform!(
    /// The function calculates the core values for updating an ellipsoid with the parallel-cut method.
    ///
    /// Arguments:
    ///
    /// * `beta1`: The parameter `beta1` represents the semi-minor axis of the ellipsoid. It is a floating-point
    /// number.
    /// * `tsq`: The parameter `tsq` represents the square of the gamma semi-axis length of the ellipsoid.
    ///
    /// ```svgbob
    ///      _.-'''''''-._
    ///    ,'      |      `.
    ///   /  |     |        \
    ///  .   |     |         .
    ///  |   |               |
    ///  |   |     .         |
    ///  |   |               |
    ///  :\  |     |        /:
    ///  | `._     |     _.' |
    ///  |   |'-.......-'    |
    ///  |   |     |         |
    /// "-τ" "-β"  0        +τ
    ///        1
    ///
    ///   2    2    2
    ///  α  = β  / τ
    ///
    ///      n    2
    ///  h = ─ ⋅ α
    ///      2
    ///             ___________
    ///            ╱ 2        2
    ///  r = h + ╲╱ h  + 1 - α
    ///
    ///        β
    ///  ϱ = ─────
    ///      r + 1
    ///
    ///        2
    ///  σ = ─────
    ///      r + 1
    ///
    ///          r
    ///  δ = ─────────
    ///      r - 1 / n
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalcCore;
    ///
    /// let ell_calc_core = EllCalcCore::new(4.0);
    /// let (rho, sigma, delta) = ell_calc_core.calc_parallel_central_cut(1.0, 4.0);
    /// assert_approx_eq!(rho, 0.4);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(delta, 1.2);
    /// ```
    )]
    pub fn calc_parallel_central_cut(&self, beta1: f64, tsq: f64) -> (f64, f64, f64) {
        let b1sq = beta1 * beta1;
        let a1sq = b1sq / tsq;
        let h = self.half_n * a1sq;
        let r = h + (1.0 - a1sq + h * h).sqrt();
        let r_plus_1 = r + 1.0;
        let rho = beta1 / r_plus_1;
        let sigma = 2.0 / r_plus_1;
        let delta = r / (r - self.inv_n);

        (rho, sigma, delta)
    }

    #[doc = svgbobdoc::transform!(
    /// The function calculates the core values needed for updating an ellipsoid with the deep-cut method.
    ///
    /// Arguments:
    ///
    /// * `beta`: The `beta` parameter represents a value used in the calculation of the core of updating
    /// the ellipsoid with the deep-cut. It is of type `f64`, which means it is a 64-bit floating-point
    /// number.
    /// * `tau`: The parameter `tau` represents the time constant of the system. It is a measure of how
    /// quickly the system responds to changes.
    /// * `eta`: The parameter `eta` represents the deep-cut factor. It is a measure of how much the
    /// ellipsoid is being updated or modified.
    ///
    /// ```svgbob
    ///       _.-'''''''-._
    ///     ,'    |        `.
    ///    /      |          \
    ///   .       |           .
    ///   |       |           |
    ///   |       | .         |
    ///   |       |           |
    ///   :\      |          /:
    ///   | `._   |       _.' |
    ///   |    '-.......-'    |
    ///   |       |           |
    ///  "-τ"     "-β"       +τ
    ///
    ///         η
    ///   ϱ = ─────
    ///       n + 1
    ///
    ///       2 ⋅ ϱ
    ///   σ = ─────
    ///       "τ + β"
    ///
    ///          2       2    2
    ///         n       τ  - β
    ///   δ = ────── ⋅  ───────
    ///        2           2
    ///       n  - 1      τ
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalcCore;
    ///
    /// let ell_calc_core = EllCalcCore::new(4.0);
    /// let (rho, sigma, delta) = ell_calc_core.calc_bias_cut_fast(1.0, 2.0, 6.0);
    /// assert_approx_eq!(rho, 1.2);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(delta, 0.8);
    /// ```
    )]
    pub fn calc_bias_cut_fast(&self, beta: f64, tau: f64, eta: f64) -> (f64, f64, f64) {
        let rho = eta / self.n_plus_1;
        let sigma = 2.0 * rho / (tau + beta);
        let alpha = beta / tau;
        let delta = self.cst1 * (1.0 - alpha * alpha);
        (rho, sigma, delta)
    }

    #[doc = svgbobdoc::transform!(
    /// The function calculates the core values needed for updating an ellipsoid with the deep-cut method.
    ///
    /// Arguments:
    ///
    /// * `beta`: The `beta` parameter represents a value used in the calculation of the core of updating
    /// the ellipsoid with the deep-cut. It is of type `f64`, which means it is a 64-bit floating-point
    /// number.
    /// * `tau`: The parameter `tau` represents the time constant of the system. It is a measure of how
    /// quickly the system responds to changes.
    /// * `eta`: The parameter `eta` represents the deep-cut factor. It is a measure of how much the
    /// ellipsoid is being updated or modified.
    ///
    /// ```svgbob
    ///       _.-'''''''-._
    ///     ,'    |        `.
    ///    /      |          \
    ///   .       |           .
    ///   |       |           |
    ///   |       | .         |
    ///   |       |           |
    ///   :\      |          /:
    ///   | `._   |       _.' |
    ///   |    '-.......-'    |
    ///   |       |           |
    ///  "-τ"     "-β"       +τ
    ///
    ///   η = τ + n ⋅ β
    ///   
    ///         η
    ///   ϱ = ─────
    ///       n + 1
    ///
    ///       2 ⋅ ϱ
    ///   σ = ─────
    ///       "τ + β"
    ///
    ///          2       2    2
    ///         n       τ  - β
    ///   δ = ────── ⋅  ───────
    ///        2           2
    ///       n  - 1      τ
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalcCore;
    ///
    /// let ell_calc_core = EllCalcCore::new(4.0);
    /// let (rho, sigma, delta) = ell_calc_core.calc_bias_cut(1.0, 2.0);
    /// assert_approx_eq!(rho, 1.2);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(delta, 0.8);
    /// ```
    )]
    #[inline]
    pub fn calc_bias_cut(&self, beta: f64, tau: f64) -> (f64, f64, f64) {
        let eta = tau + self.n_f * beta;
        self.calc_bias_cut_fast(beta, tau, eta)
    }

    #[doc = svgbobdoc::transform!(
    /// The `calc_central_cut_core` function calculates the core values needed for updating an ellipsoid with the
    /// central-cut.
    ///
    /// Arguments:
    ///
    /// * `tsq`: The parameter `tsq` represents the square of the time taken to update the ellipsoid with
    /// the central-cut.
    ///
    /// ```svgbob
    ///       _.-'''''''-._
    ///     ,'      |      `.
    ///    /        |        \
    ///   .         |         .
    ///   |                   |
    ///   |         .         |
    ///   |                   |
    ///   :\        |        /:
    ///   | `._     |     _.' |
    ///   |    '-.......-'    |
    ///   |         |         |
    ///  "-τ"       0        +τ
    ///
    ///         2
    ///   σ = ─────
    ///       n + 1
    ///
    ///         τ
    ///   ϱ = ─────
    ///       n + 1
    ///
    ///          2
    ///         n
    ///   δ = ──────
    ///        2
    ///       n  - 1
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalcCore;
    /// let ell_calc_core = EllCalcCore::new(4.0);
    /// let (rho, sigma, delta) = ell_calc_core.calc_central_cut(4.0);
    /// assert_approx_eq!(rho, 0.4);
    /// assert_approx_eq!(sigma, 0.4);
    /// assert_approx_eq!(delta, 16.0/15.0);
    /// ```
    )]
    pub fn calc_central_cut(&self, tsq: f64) -> (f64, f64, f64) {
        // self.mu = self.half_n_minus_1;
        let sigma = self.cst2;
        let rho = tsq.sqrt() / self.n_plus_1;
        let delta = self.cst1;
        (rho, sigma, delta)
    }
}

/// The `EllCalc` struct represents an ellipsoid search space in Rust.
///
///  EllCalc = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
///
/// Properties:
///
/// * `n_f`: The `n_f` property is a floating-point number that represents the dimensionality of the
/// search space. It indicates the number of variables or dimensions in the ellipsoid search space.
/// * `helper`: The `helper` property is of type `EllCalcCore` and is used to perform
/// calculations related to the ellipsoid search space. It is a separate struct that contains the
/// necessary methods and data for these calculations.
/// * `use_parallel_cut`: A boolean flag indicating whether to use parallel cut or not.
#[derive(Debug, Clone)]
pub struct EllCalc {
    n_f: f64,
    pub helper: EllCalcCore,
    pub use_parallel_cut: bool,
}

impl EllCalc {
    /// The `new` function constructs a new [`EllCalc`] object with a given value for `n_f` and sets the
    /// `use_parallel_cut` flag to `true`.
    ///
    /// Arguments:
    ///
    /// * `n_f`: The parameter `n_f` is a floating-point number that is used to initialize the `EllCalc`
    /// struct.
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the `EllCalc` struct.
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use ellalgo_rs::ell_calc::EllCalc;
    /// let ell_calc = EllCalc::new(4);
    /// assert!(ell_calc.use_parallel_cut);
    /// ```
    pub fn new(n: usize) -> EllCalc {
        let n_f = n as f64;
        let helper = EllCalcCore::new(n_f);

        EllCalc {
            n_f,
            helper,
            use_parallel_cut: true,
        }
    }

    // pub fn update_cut(&mut self, beta: f64) -> CutStatus { self.calc_bias_cut(beta) }

    /// The function calculates the updating of an ellipsoid with a single or parallel-cut.
    ///
    /// Arguments:
    ///
    /// * `beta`: The `beta` parameter is a tuple containing two values: `beta0` and `beta1_opt`. `beta0` is of type
    /// `f64` and `beta1_opt` is an optional value of type `Option<f64>`.
    /// * `tsq`: The `tsq` parameter is a reference to a `f64` value.
    pub fn calc_single_or_parallel_bias_cut(
        &self,
        beta: &(f64, Option<f64>),
        tsq: f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        let (beta0, beta1_opt) = *beta;
        if let Some(beta1) = beta1_opt {
            self.calc_parallel_bias_cut(beta0, beta1, tsq)
        } else {
            self.calc_bias_cut(beta0, tsq)
        }
    }

    /// The function calculates the updating of an ellipsoid with a single or parallel-cut (one of them is central-cut).
    ///
    /// Arguments:
    ///
    /// * `beta`: The `beta` parameter is a tuple containing two values: `f64` and `Option<f64>`.
    /// The first value, denoted as `_b0`, is of type `f64`. The second value, `beta1_opt`, is of type `Option<f64>`.
    /// * `tsq`: The `tsq` parameter is a reference to a `f64` value.
    pub fn calc_single_or_parallel_central_cut(
        &self,
        beta: &(f64, Option<f64>),
        tsq: f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        let (_b0, beta1_opt) = *beta;
        if let Some(beta1) = beta1_opt {
            self.calc_parallel_central_cut(beta1, tsq)
        } else {
            self.calc_central_cut(tsq)
        }
    }

    /// The function calculates the updating of an ellipsoid with a single or parallel-cut (discrete version).
    ///
    /// Arguments:
    ///
    /// * `beta`: The `beta` parameter is a tuple containing two values: `beta0` and `beta1_opt`. `beta0` is of type
    /// `f64` and `beta1_opt` is an optional value of type `Option<f64>`.
    /// * `tsq`: The `tsq` parameter is a reference to a `f64` value.
    pub fn calc_single_or_parallel_q(
        &self,
        beta: &(f64, Option<f64>),
        tsq: f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        let (beta0, beta1_opt) = *beta;
        if let Some(beta1) = beta1_opt {
            self.calc_parallel_q(beta0, beta1, tsq)
        } else {
            self.calc_bias_cut_q(beta0, tsq)
        }
    }

    /// Parallel Deep Cut
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalc;
    /// use ellalgo_rs::cutting_plane::CutStatus;
    ///
    /// let ell_calc = EllCalc::new(4);
    /// let (status, _result) = ell_calc.calc_parallel_bias_cut(0.07, 0.03, 0.01);
    /// assert_eq!(status, CutStatus::NoSoln);
    ///
    /// let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_bias_cut(0.0, 0.05, 0.01);
    /// assert_eq!(status, CutStatus::Success);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(rho, 0.02);
    /// assert_approx_eq!(delta, 1.2);
    ///
    /// let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_bias_cut(0.05, 0.11, 0.01);
    /// assert_eq!(status, CutStatus::Success);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(rho, 0.06);
    /// assert_approx_eq!(delta, 0.8);
    ///
    /// let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
    /// assert_eq!(status, CutStatus::Success);
    /// assert_approx_eq!(sigma, 0.928);
    /// assert_approx_eq!(rho, 0.0232);
    /// assert_approx_eq!(delta, 1.232);
    /// ```
    pub fn calc_parallel_bias_cut(
        &self,
        beta0: f64,
        beta1: f64,
        tsq: f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        if beta1 < beta0 {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        let b1sqn = beta1 * (beta1 / tsq);
        let t1n = 1.0 - b1sqn;
        if t1n < 0.0 || !self.use_parallel_cut {
            return self.calc_bias_cut(beta0, tsq);
        }

        (
            CutStatus::Success,
            self.helper.calc_parallel_bias_cut(beta0, beta1, tsq),
        )
    }

    /// Discrete Parallel Deep Cut
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalc;
    /// use ellalgo_rs::cutting_plane::CutStatus;
    ///
    /// let ell_calc = EllCalc::new(4);
    /// let (status, _result) = ell_calc.calc_parallel_q(-0.07, 0.07, 0.01);
    /// assert_eq!(status, CutStatus::NoEffect);
    ///
    /// let (status, _result) = ell_calc.calc_parallel_q(-0.04, 0.0625, 0.01);
    /// assert_eq!(status, CutStatus::NoEffect);
    /// ```
    pub fn calc_parallel_q(
        &self,
        beta0: f64,
        beta1: f64,
        tsq: f64,
    ) -> (CutStatus, (f64, f64, f64)) {
        if beta1 < beta0 {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        if (beta1 > 0.0) && beta1 * beta1 >= tsq || !self.use_parallel_cut {
            return self.calc_bias_cut_q(beta0, tsq);
        }

        let b0b1 = beta0 * beta1;
        let eta = tsq + self.n_f * b0b1;
        if eta <= 0.0 {
            return (CutStatus::NoEffect, (0.0, 0.0, 1.0)); // no effect
        }

        (
            CutStatus::Success,
            self.helper
                .calc_parallel_bias_cut_fast(beta0, beta1, tsq, b0b1, eta),
        )
    }

    /// Parallel Central Cut
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalc;
    /// use ellalgo_rs::cutting_plane::CutStatus;
    ///
    /// let ell_calc = EllCalc::new(4);
    /// let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(0.11, 0.01);
    /// assert_eq!(status, CutStatus::Success);
    /// assert_approx_eq!(sigma, 0.4);
    /// assert_approx_eq!(rho, 0.02);
    /// assert_approx_eq!(delta, 16.0 / 15.0);
    ///
    /// let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(0.05, 0.01);
    /// assert_eq!(status, CutStatus::Success);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(rho, 0.02);
    /// assert_approx_eq!(delta, 1.2);
    /// ```
    pub fn calc_parallel_central_cut(&self, beta1: f64, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        if beta1 < 0.0 {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no solution
        }
        let b1sqn = beta1 * (beta1 / tsq);
        let t1n = 1.0 - b1sqn;
        if t1n < 0.0 || !self.use_parallel_cut {
            return self.calc_central_cut(tsq);
        }
        (
            CutStatus::Success,
            self.helper.calc_parallel_central_cut(beta1, tsq),
        )
    }

    /// Deep Cut
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalc;
    /// use ellalgo_rs::cutting_plane::CutStatus;
    ///
    /// let ell_calc = EllCalc::new(4);
    /// let (status, _result) = ell_calc.calc_bias_cut(0.11, 0.01);
    /// assert_eq!(status, CutStatus::NoSoln);
    /// let (status, _result) = ell_calc.calc_bias_cut(0.0, 0.01);
    /// assert_eq!(status, CutStatus::Success);
    ///
    /// let (status, (rho, sigma, delta)) = ell_calc.calc_bias_cut(0.05, 0.01);
    /// assert_eq!(status, CutStatus::Success);
    /// assert_approx_eq!(sigma, 0.8);
    /// assert_approx_eq!(rho, 0.06);
    /// assert_approx_eq!(delta, 0.8);
    /// ```
    pub fn calc_bias_cut(&self, beta: f64, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        if tsq < beta * beta {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        let tau = tsq.sqrt();
        (CutStatus::Success, self.helper.calc_bias_cut(beta, tau))
    }

    /// Discrete Deep Cut
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalc;
    /// use ellalgo_rs::cutting_plane::CutStatus;
    ///
    /// let ell_calc = EllCalc::new(4);
    /// let (status, _result) = ell_calc.calc_bias_cut_q(-0.05, 0.01);
    /// assert_eq!(status, CutStatus::NoEffect);
    /// ```
    pub fn calc_bias_cut_q(&self, beta: f64, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        let tau = tsq.sqrt();

        if tau < beta {
            return (CutStatus::NoSoln, (0.0, 0.0, 0.0)); // no sol'n
        }

        let eta = tau + self.n_f * beta;
        if eta < 0.0 {
            return (CutStatus::NoEffect, (0.0, 0.0, 1.0)); // no effect
        }

        (
            CutStatus::Success,
            self.helper.calc_bias_cut_fast(beta, tau, eta),
        )
    }

    /// Central Cut
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use approx_eq::assert_approx_eq;
    /// use ellalgo_rs::ell_calc::EllCalc;
    /// use ellalgo_rs::cutting_plane::CutStatus;
    ///
    /// let ell_calc = EllCalc::new(4);
    /// let (status, (rho, sigma, delta)) = ell_calc.calc_central_cut(0.01);
    ///
    /// assert_eq!(status, CutStatus::Success);
    /// assert_approx_eq!(sigma, 0.4);
    /// assert_approx_eq!(rho, 0.02);
    /// assert_approx_eq!(delta, 16.0 / 15.0);
    /// ```
    #[inline]
    pub fn calc_central_cut(&self, tsq: f64) -> (CutStatus, (f64, f64, f64)) {
        // self.mu = self.half_n_minus_1;
        (CutStatus::Success, self.helper.calc_central_cut(tsq))
    }
}

// pub trait UpdateByCutChoice {
//     fn update_by(self, ell: &mut EllCalc) -> CutStatus;
// }
#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    pub fn test_construct() {
        let helper = EllCalcCore::new(4.0);
        assert_eq!(helper.n_f, 4.0);
        assert_eq!(helper.half_n, 2.0);
        assert_approx_eq!(helper.cst1, 16.0 / 15.0);
        assert_approx_eq!(helper.cst2, 0.4);
    }

    #[test]
    pub fn test_calc_parallel_bias_cut_fast() {
        let ell_calc_core = EllCalcCore::new(4.0);
        let (rho, sigma, delta) =
            ell_calc_core.calc_parallel_bias_cut_fast(1.0, 2.0, 4.0, 2.0, 12.0);
        assert_approx_eq!(rho, 1.2);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 0.8);
    }

    #[test]
    pub fn test_calc_bias_cut_fast() {
        let ell_calc_core = EllCalcCore::new(4.0);
        let (rho, sigma, delta) = ell_calc_core.calc_bias_cut_fast(1.0, 2.0, 6.0);
        assert_approx_eq!(rho, 1.2);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 0.8);
    }

    #[test]
    pub fn test_calc_central_cut() {
        let ell_calc = EllCalc::new(4);
        // ell_calc.tsq = 0.01;
        let (status, (rho, sigma, delta)) = ell_calc.calc_central_cut(0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.4);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 16.0 / 15.0);
    }

    #[test]
    pub fn test_calc_bias_cut() {
        let ell_calc = EllCalc::new(4);
        // ell_calc.tsq = 0.01;
        let (status, _result) = ell_calc.calc_bias_cut(0.11, 0.01);
        assert_eq!(status, CutStatus::NoSoln);
        let (status, _result) = ell_calc.calc_bias_cut(0.0, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (status, _result) = ell_calc.calc_bias_cut_q(-0.05, 0.01);
        assert_eq!(status, CutStatus::NoEffect);

        // ell_calc.tsq = 0.01;
        let (status, (rho, sigma, delta)) = ell_calc.calc_bias_cut(0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(delta, 0.8);
    }

    #[test]
    pub fn test_calc_parallel_central_cut() {
        let ell_calc = EllCalc::new(4);
        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(0.11, 0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.4);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 16.0 / 15.0);

        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_central_cut(0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 1.2);
    }

    #[test]
    pub fn test_calc_parallel() {
        let ell_calc = EllCalc::new(4);
        // ell_calc.tsq = 0.01;
        let (status, _result) = ell_calc.calc_parallel_bias_cut(0.07, 0.03, 0.01);
        assert_eq!(status, CutStatus::NoSoln);

        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_bias_cut(0.0, 0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 1.2);

        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_bias_cut(0.05, 0.11, 0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(delta, 0.8);

        let (status, _result) = ell_calc.calc_parallel_q(-0.07, 0.07, 0.01);
        assert_eq!(status, CutStatus::NoEffect);

        let (status, (rho, sigma, delta)) = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
        assert_eq!(status, CutStatus::Success);
        assert_approx_eq!(sigma, 0.928);
        assert_approx_eq!(rho, 0.0232);
        assert_approx_eq!(delta, 1.232);

        let (status, _result) = ell_calc.calc_parallel_q(-0.04, 0.0625, 0.01);
        assert_eq!(status, CutStatus::NoEffect);
    }

    #[test]
    fn test_construct2() {
        let ell_calc = EllCalc::new(4);
        assert!(ell_calc.use_parallel_cut);
        assert_eq!(ell_calc.n_f, 4.0);
    }

    #[test]
    fn test_calc_central_cut2() {
        let ell_calc = EllCalc::new(4);
        let (status, result) =
            ell_calc.calc_single_or_parallel_central_cut(&(0.0, Some(0.05)), 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(delta, 1.2);
    }

    #[test]
    fn test_calc_bias_cut2() {
        let ell_calc = EllCalc::new(4);
        let (status, _result) = ell_calc.calc_bias_cut(0.11, 0.01);
        assert_eq!(status, CutStatus::NoSoln);
        let (status, _result) = ell_calc.calc_bias_cut(0.01, 0.01);
        assert_eq!(status, CutStatus::Success);

        let (status, result) = ell_calc.calc_bias_cut(0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 0.8);
    }

    #[test]
    fn test_calc_parallel_central_cut2() {
        let ell_calc = EllCalc::new(4);
        let (status, result) =
            ell_calc.calc_single_or_parallel_central_cut(&(0.0, Some(0.05)), 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 1.2);
    }

    #[test]
    fn test_calc_parallel2() {
        let ell_calc = EllCalc::new(4);
        let (status, _result) = ell_calc.calc_parallel_bias_cut(0.07, 0.03, 0.01);
        assert_eq!(status, CutStatus::NoSoln);
        let (status, result) = ell_calc.calc_parallel_bias_cut(0.0, 0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 1.2);
        let (status, result) = ell_calc.calc_parallel_bias_cut(0.05, 0.11, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 0.8);
        let (status, result) = ell_calc.calc_parallel_bias_cut(0.01, 0.04, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.0232);
        assert_approx_eq!(sigma, 0.928);
        assert_approx_eq!(delta, 1.232);
    }

    #[test]
    fn test_calc_bias_cut_q2() {
        let ell_calc_q = EllCalc::new(4);
        let (status, _result) = ell_calc_q.calc_bias_cut_q(0.11, 0.01);
        assert_eq!(status, CutStatus::NoSoln);
        let (status, _result) = ell_calc_q.calc_bias_cut_q(0.01, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (status, _result) = ell_calc_q.calc_bias_cut_q(-0.05, 0.01);
        assert_eq!(status, CutStatus::NoEffect);
        let (status, result) = ell_calc_q.calc_bias_cut_q(0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 0.8);
    }

    #[test]
    fn test_calc_parallel_q2() {
        let ell_calc = EllCalc::new(4);
        let (status, _result) = ell_calc.calc_parallel_q(0.07, 0.03, 0.01);
        assert_eq!(status, CutStatus::NoSoln);
        let (status, _result) = ell_calc.calc_parallel_q(-0.04, 0.0625, 0.01);
        assert_eq!(status, CutStatus::NoEffect);
        let (status, result) = ell_calc.calc_parallel_q(0.0, 0.05, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.02);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 1.2);
        let (status, result) = ell_calc.calc_parallel_q(0.05, 0.11, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.06);
        assert_approx_eq!(sigma, 0.8);
        assert_approx_eq!(delta, 0.8);
        let (status, result) = ell_calc.calc_parallel_q(0.01, 0.04, 0.01);
        assert_eq!(status, CutStatus::Success);
        let (rho, sigma, delta) = result;
        assert_approx_eq!(rho, 0.0232);
        assert_approx_eq!(sigma, 0.928);
        assert_approx_eq!(delta, 1.232);
    }
}
