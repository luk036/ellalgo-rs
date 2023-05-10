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
    pub tol: f64,         // error tolerrance
}

type CInfo = (bool, usize);

pub trait UpdateByCutChoices<SS> {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    fn update_dc_by(&self, space: &mut SS, grad: &Self::ArrayType) -> CutStatus;
    fn update_cc_by(&self, space: &mut SS, grad: &Self::ArrayType) -> CutStatus;
    fn update_q_by(&self, space: &mut SS, grad: &Self::ArrayType) -> CutStatus;
}

/// Oracle for feasibility problems
pub trait OracleFeas<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_feas(&mut self, xc: &ArrayType) -> Option<(ArrayType, Self::CutChoices)>;
}

/// Oracle for optimization problems
pub trait OracleOptim<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64>) for parallel cut
    fn assess_optim(
        &mut self,
        xc: &ArrayType,
        tea: &mut f64,
    ) -> ((ArrayType, Self::CutChoices), bool);
}

/// Oracle for quantized optimization problems
pub trait OracleOptimQ<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_optim_q(
        &mut self,
        xc: &ArrayType,
        tea: &mut f64,
        retry: bool,
    ) -> ((ArrayType, Self::CutChoices), bool, ArrayType, bool);
}

/// Oracle for binary search
pub trait OracleBS {
    fn assess_bs(&mut self, tea: f64) -> bool;
}

pub trait SearchSpace {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general

    fn xc(&self) -> Self::ArrayType;

    fn tsq(&self) -> f64; // measure of the search space

    fn update_dc<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
        Self: Sized;

    fn update_cc<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
    where
        T: UpdateByCutChoices<Self, ArrayType = Self::ArrayType>,
        Self: Sized;
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

/**
 * @brief Find a point in a convex set (defined through a cutting-plane oracle).
 *
 * A function f(x) is *convex* if there always exist a g(x)
 * such that f(z) >= f(x) + g(x)^T * (z - x), forall z, x in dom f.
 * Note that dom f does not need to be a convex set in our definition.
 * The affine function g^T (x - xc) + beta is called a cutting-plane,
 * or a "cut" for short.
 * This algorithm solves the following feasibility problem:
 *
 *   find x
 *   s.t. f(x) <= 0,
 *
 * A *separation oracle* asserts that an evalution point x0 is feasible,
 * or provide a cut that separates the feasible region and x0.
 *
 * @tparam Oracle
 * @tparam Space
 * @param omega perform assessment on x0
 * @param space search Space containing x*
 * @param options maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_feas<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    options: &Options,
) -> CInfo
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleFeas<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace,
{
    for niter in 0..options.max_iters {
        let cut = omega.assess_feas(&space.xc()); // query the oracle at &space.xc()
        if cut.is_none() {
            // feasible sol'n obtained
            return (true, niter);
        }
        let status = space.update_dc::<T>(&cut.unwrap()); // update space
        if status != CutStatus::Success || space.tsq() < options.tol {
            return (false, niter);
        }
    }
    (false, options.max_iters)
}

/**
 * @brief Cutting-plane method for solving convex problem
 *
 * @tparam Oracle
 * @tparam Space
 * @tparam Num
 * @param omega perform assessment on x0
 * @param space search Space containing x*
 * @param tea best-so-far optimal sol'n
 * @param options maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_optim<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    tea: &mut f64,
    options: &Options,
) -> (Option<Space::ArrayType>, usize)
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleOptim<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace,
{
    let mut x_best: Option<Space::ArrayType> = None;

    for niter in 0..options.max_iters {
        let (cut, shrunk) = omega.assess_optim(&space.xc(), tea); // query the oracle at &space.xc()
        let status = if shrunk {
            // better tea obtained
            x_best = Some(space.xc());
            space.update_dc::<T>(&cut) // update space
        } else {
            space.update_cc::<T>(&cut) // update space
        };
        if status != CutStatus::Success || space.tsq() < options.tol {
            return (x_best, niter);
        }
    }
    (x_best, options.max_iters)
} // END

/**
 * @brief Cutting-plane method for solving convex discrete optimization problem
 *
 * @tparam Oracle
 * @tparam Space
 * @param omega perform assessment on x0
 * @param space search Space containing x*
 * @param tea best-so-far optimal sol'n
 * @param options maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_optim_q<T, Oracle, Space>(
    omega: &mut Oracle,
    space_q: &mut Space,
    tea: &mut f64,
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
        let (cut, shrunk, x_q, more_alt) = omega.assess_optim_q(&space_q.xc(), tea, retry); // query the oracle at &space.xc()
        if shrunk {
            // best tea obtained
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
        if space_q.tsq() < options.tol {
            return (x_best, niter);
        }
    }
    (x_best, options.max_iters)
} // END

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 * @param omega    perform assessment on x0
 * @param I        interval containing x*
 * @param     options  maximum iteration and error tolerance etc.
 * @return CInfo
 */
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
        if tau < options.tol {
            return (upper != u_orig, niter);
        }
        let mut tea = lower; // l may be `i32` or `Fraction`
        tea += tau;
        if omega.assess_bs(tea) {
            // feasible sol'n obtained
            upper = tea;
        } else {
            lower = tea;
        }
    }
    (upper != u_orig, options.max_iters)
}

// /**
//  * @brief
//  *
//  * @tparam Oracle
//  * @tparam Space
//  */
// template <typename Oracle, typename Space>  //
// class bsearch_adaptor {
//   private:
//     Oracle& _omega;
//     Space& _S;
//     const Options _options;

//   public:
//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param omega perform assessment on x0
//      * @param space search Space containing x*
//      */
//     bsearch_adaptor(Oracle& omega, Space& space) : bsearch_adaptor{omega, space, Options()} {}

//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param omega perform assessment on x0
//      * @param space search Space containing x*
//      * @param options maximum iteration and error tolerance etc.
//      */
//     bsearch_adaptor(Oracle& omega, Space& space, const Options& options)
//         : _omega{omega}, _S{space}, _options{options} {}

//     /**
//      * @brief get best x
//      *
//      * @return auto
//      */
//     let mut x_best() const { return self.&space.xc(); }

//     /**
//      * @brief
//      *
//      * @param tea the best-so-far optimal value
//      * @return bool
//      */
//     template <typename Num> let mut operator()(const Num& tea) -> bool {
//         Space space = self.space.copy();
//         self.omega.update(tea);
//         let ell_info = cutting_plane_feas(self.omega, space, self.options);
//         if ell_info.feasible {
//             self.space.set_xc(&space.xc());
//         }
//         return ell_info.feasible;
//     }
// };
