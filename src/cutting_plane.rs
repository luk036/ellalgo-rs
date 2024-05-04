#![allow(non_snake_case)]

#[derive(Debug, PartialEq, Eq)]
pub enum CutStatus {
    Success,
    NoSoln,
    NoEffect,
    Unknown,
}

pub struct Options {
    pub max_iters: usize, // maximum number of iterations
    pub tolerance: f64,   // error tolerrance
}

impl Default for Options {
    fn default() -> Options {
        Options {
            max_iters: 2000,
            tolerance: 1e-20,
        }
    }
}

type CInfo = (bool, usize);

pub trait UpdateByCutChoices<SearchSpace> {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    fn update_bias_cut_by(&self, space: &mut SearchSpace, grad: &Self::ArrayType) -> CutStatus;
    fn update_central_cut_by(&self, space: &mut SearchSpace, grad: &Self::ArrayType) -> CutStatus;
    fn update_q_by(&self, space: &mut SearchSpace, grad: &Self::ArrayType) -> CutStatus;
}

/// Oracle for feasibility problems
pub trait OracleFeas<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_feas(&mut self, xc: &ArrayType) -> Option<(ArrayType, Self::CutChoices)>;
}

/// Oracle for feasibility problems
pub trait OracleFeas2<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_feas(&mut self, xc: &ArrayType) -> Option<(ArrayType, Self::CutChoices)>;
    fn update(&mut self, gamma: f64);
}

/// Oracle for optimization problems
pub trait OracleOptim<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64>) for parallel cut
    fn assess_optim(
        &mut self,
        xc: &ArrayType,
        gamma: &mut f64,
    ) -> ((ArrayType, Self::CutChoices), bool);
}

/// Oracle for quantized optimization problems
pub trait OracleOptimQ<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_optim_q(
        &mut self,
        xc: &ArrayType,
        gamma: &mut f64,
        retry: bool,
    ) -> ((ArrayType, Self::CutChoices), bool, ArrayType, bool);
}

/// Oracle for binary search
pub trait OracleBS {
    fn assess_bs(&mut self, gamma: f64) -> bool;
}

pub trait SearchSpace {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general

    fn xc(&self) -> Self::ArrayType;

    fn tsq(&self) -> f64; // measure of the search space

    fn update_bias_cut<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
        Self: Sized;

    fn update_central_cut<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
        Self: Sized;

    fn set_xc(&mut self, x: Self::ArrayType);
}

pub trait SearchSpaceQ {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general

    fn xc(&self) -> Self::ArrayType;

    fn tsq(&self) -> f64; // measure of the search space

    fn update_q<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
        Self: Sized;
}

/// The function `cutting_plane_feas` iteratively updates a search space using a cutting plane oracle
/// until a feasible solution is found or the maximum number of iterations is reached.
///
/// Arguments:
///
/// * `omega`: `omega` is an instance of the `Oracle` trait, which represents an oracle that provides
/// information about the feasibility of a solution. The `Oracle` trait has a method `assess_feas` that
/// takes a reference to the current solution `&space.xc()` and returns an optional `
/// * `space`: The `space` parameter represents the search space in which the optimization problem is
/// being solved. It is a mutable reference to an object that implements the `SearchSpace` trait.
/// * `options`: The `options` parameter is of type `Options` and contains various settings for the
/// cutting plane algorithm. It likely includes properties such as `max_iters` (maximum number of
/// iterations), `` (tolerance for termination), and other parameters that control the behavior of
/// the algorithm.
///
/// Returns:
///
/// The function `cutting_plane_feas` returns a tuple `(bool, usize)`. The first element of the tuple
/// represents whether a feasible solution was obtained (`true` if yes, `false` if no), and the second
/// element represents the number of iterations performed.
#[allow(dead_code)]
pub fn cutting_plane_feas<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    options: &Options,
) -> (Option<Space::ArrayType>, usize)
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleFeas<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace,
{
    for niter in 0..options.max_iters {
        let cut = omega.assess_feas(&space.xc()); // query the oracle at &space.xc()
        if cut.is_none() {
            // feasible sol'n obtained
            return (Some(space.xc()), niter);
        }
        let status = space.update_bias_cut::<T>(&cut.unwrap()); // update space
        if status != CutStatus::Success || space.tsq() < options.tolerance {
            return (None, niter);
        }
    }
    (None, options.max_iters)
}

/// The function `cutting_plane_feas` iteratively updates a search space using a cutting plane oracle
/// until a feasible solution is found or the maximum number of iterations is reached.
///
/// Arguments:
///
/// * `omega`: `omega` is an instance of the `Oracle` trait, which represents an oracle that provides
/// information about the feasibility of a solution. The `Oracle` trait has a method `assess_feas` that
/// takes a reference to the current solution `&space.xc()` and returns an optional `
/// * `space`: The `space` parameter represents the search space in which the optimization problem is
/// being solved. It is a mutable reference to an object that implements the `SearchSpace` trait.
/// * `options`: The `options` parameter is of type `Options` and contains various settings for the
/// cutting plane algorithm. It likely includes properties such as `max_iters` (maximum number of
/// iterations), `` (tolerance for termination), and other parameters that control the behavior of
/// the algorithm.
///
/// Returns:
///
/// The function `cutting_plane_feas` returns a tuple `(bool, usize)`. The first element of the tuple
/// represents whether a feasible solution was obtained (`true` if yes, `false` if no), and the second
/// element represents the number of iterations performed.
#[allow(dead_code)]
pub fn cutting_plane_feas2<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    options: &Options,
) -> (Option<Space::ArrayType>, usize)
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleFeas2<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace,
{
    for niter in 0..options.max_iters {
        let cut = omega.assess_feas(&space.xc()); // query the oracle at &space.xc()
        if cut.is_none() {
            // feasible sol'n obtained
            return (Some(space.xc()), niter);
        }
        let status = space.update_bias_cut::<T>(&cut.unwrap()); // update space
        if status != CutStatus::Success || space.tsq() < options.tolerance {
            return (None, niter);
        }
    }
    (None, options.max_iters)
}

/// The function `cutting_plane_optim` performs cutting plane optimization on a given search space using
/// an oracle.
///
/// Arguments:
///
/// * `omega`: The `omega` parameter is an instance of the `Oracle` trait, which represents an
/// optimization oracle. The oracle provides information about the optimization problem, such as the
/// objective function and constraints.
/// * `space`: The `space` parameter represents the search space, which is a type that implements the
/// `SearchSpace` trait. It contains the current state of the optimization problem, including the
/// decision variables and any additional information needed for the optimization algorithm.
/// * `gamma`: The parameter `gamma` represents the current value of the gamma function that the
/// optimization algorithm is trying to minimize.
/// * `options`: The `options` parameter is of type `Options` and contains various settings for the
/// optimization algorithm. It likely includes parameters such as the maximum number of iterations
/// (`max_iters`) and the tolerance (``) for convergence.
#[allow(dead_code)]
pub fn cutting_plane_optim<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    gamma: &mut f64,
    options: &Options,
) -> (Option<Space::ArrayType>, usize)
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleOptim<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace,
{
    let mut x_best: Option<Space::ArrayType> = None;

    for niter in 0..options.max_iters {
        let (cut, shrunk) = omega.assess_optim(&space.xc(), gamma); // query the oracle at &space.xc()
        let status = if shrunk {
            // better gamma obtained
            x_best = Some(space.xc());
            space.update_central_cut::<T>(&cut) // update space
        } else {
            space.update_bias_cut::<T>(&cut) // update space
        };
        if status != CutStatus::Success || space.tsq() < options.tolerance {
            return (x_best, niter);
        }
    }
    (x_best, options.max_iters)
} // END

/// The function implements the cutting-plane method for solving a convex discrete optimization problem.
///
/// Arguments:
///
/// * `omega`: The parameter "omega" is an instance of the OracleOptimQ trait, which represents an
/// oracle that provides assessments for the cutting-plane method. It is used to query the oracle for
/// assessments on the current solution.
/// * `space_q`: The parameter `space_q` is a mutable reference to a `Space` object, which represents
/// the search space containing the optimal solution `x*`. It is used to update the space based on the
/// cuts obtained from the oracle.
/// * `gamma`: The parameter "gamma" represents the best-so-far optimal solution. It is a mutable reference
/// to a floating-point number (f64).
/// * `options`: The `options` parameter is a struct that contains various options for the cutting-plane
/// method. It includes parameters such as the maximum number of iterations (`max_iters`) and the error
/// tolerance (``). These options control the termination criteria for the method.
#[allow(dead_code)]
pub fn cutting_plane_optim_q<T, Oracle, Space>(
    omega: &mut Oracle,
    space_q: &mut Space,
    gamma: &mut f64,
    options: &Options,
) -> (Option<Space::ArrayType>, usize)
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleOptimQ<Space::ArrayType, CutChoices = T>,
    Space: SearchSpaceQ,
{
    let mut x_best: Option<Space::ArrayType> = None;
    let mut retry = false;

    for niter in 0..options.max_iters {
        let (cut, shrunk, x_q, more_alt) = omega.assess_optim_q(&space_q.xc(), gamma, retry); // query the oracle at &space.xc()
        if shrunk {
            // best gamma obtained
            x_best = Some(x_q);
            retry = false;
        }
        let status = space_q.update_q::<T>(&cut); // update space
        match &status {
            CutStatus::Success => {
                retry = false;
            }
            CutStatus::NoSoln => {
                return (x_best, niter);
            }
            CutStatus::NoEffect => {
                if !more_alt {
                    // no more alternative cut
                    return (x_best, niter);
                }
                retry = true;
            }
            _ => {}
        }
        if space_q.tsq() < options.tolerance {
            return (x_best, niter);
        }
    }
    (x_best, options.max_iters)
} // END

pub struct BSearchAdaptor<T, Oracle, Space>
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleFeas2<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace,
{
    pub omega: Oracle,
    pub space: Space,
    pub options: Options,
}

#[allow(dead_code)]
impl<T, Oracle, Space> BSearchAdaptor<T, Oracle, Space>
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleFeas2<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace + Clone,
{
    pub fn new(omega: Oracle, space: Space, options: Options) -> Self {
        BSearchAdaptor { omega, space, options }
    }
}

impl<T, Oracle, Space> OracleBS for BSearchAdaptor<T, Oracle, Space>
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleFeas2<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace + Clone,
{
    fn assess_bs(&mut self, gamma: f64) -> bool {
        let mut space = self.space.clone();
        self.omega.update(gamma);
        let (x_feas, _) = cutting_plane_feas2(&mut self.omega, &mut space, &self.options);
        if let Some(x) = x_feas {
            self.space.set_xc(x);
            return true;
        }
        false
    }
}

/// The `bsearch` function performs a binary search to find a feasible solution within a given interval.
///
/// Arguments:
///
/// * `omega`: The parameter `omega` is an instance of the `Oracle` trait, which is used to perform
/// assessments on a value `x0`. The specific implementation of the `Oracle` trait is not provided in
/// the code snippet, so it could be any type that implements the necessary methods for the binary
/// search
/// * `intrvl`: The `intrvl` parameter is an interval containing the gamma value `x*`. It is
/// represented as a mutable reference to a tuple `(f64, f64)`, where the first element is the lower
/// bound of the interval and the second element is the upper bound of the interval.
/// * `options`: The `options` parameter is a struct that contains various options for the binary search
/// algorithm. It includes properties such as the maximum number of iterations (`max_iters`) and the
/// error tolerance (``). These options control the termination criteria for the algorithm.
///
/// Returns:
///
/// The function `bsearch` returns a tuple of two values: a boolean indicating whether a feasible
/// solution was obtained, and the number of iterations performed.
#[allow(dead_code)]
pub fn bsearch<Oracle>(omega: &mut Oracle, intrvl: &mut (f64, f64), options: &Options) -> CInfo
where
    Oracle: OracleBS,
{
    // assume monotone
    // auto& [lower, upper] = I;
    let &mut (mut lower, mut upper) = intrvl;
    assert!(lower <= upper);
    let u_orig = upper;

    for niter in 0..options.max_iters {
        let tau = (upper - lower) / 2.0;
        if tau < options.tolerance {
            return (upper != u_orig, niter);
        }
        let mut gamma = lower; // l may be `i32` or `Fraction`
        gamma += tau;
        if omega.assess_bs(gamma) {
            // feasible sol'n obtained
            upper = gamma;
        } else {
            lower = gamma;
        }
    }
    (upper != u_orig, options.max_iters)
}
