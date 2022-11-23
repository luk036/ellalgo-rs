#[derive(Debug, PartialEq, Eq)]
pub enum CutStatus {
    Success,
    NoSoln,
    NoEffect,
    SmallEnough,
    Unknown,
}

pub struct Options {
    pub max_iter: usize,
    pub tol: f64,
}

type CInfo = (bool, usize, CutStatus);

/// TODO: support 1D problems

pub trait UpdateByCutChoices<SS> {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    fn update_by(&self, ss: &mut SS, grad: &Self::ArrayType) -> (CutStatus, f64);
}

/// Oracle for feasibility problems
pub trait OracleFeas {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_feas(&mut self, x: &Self::ArrayType) -> Option<(Self::ArrayType, Self::CutChoices)>;
}

/// Oracle for optimization problems
pub trait OracleOptim {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    type CutChoices; // f64 for single cut; (f64, Option<f64>) for parallel cut
    fn assess_optim(
        &mut self,
        x: &Self::ArrayType,
        t: &mut f64,
    ) -> ((Self::ArrayType, Self::CutChoices), bool);
}

/// Oracle for quantized optimization problems
pub trait OracleQ {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_q(
        &mut self,
        x: &Self::ArrayType,
        t: &mut f64,
        retry: bool,
    ) -> (
        (Self::ArrayType, Self::CutChoices),
        bool,
        Self::ArrayType,
        bool,
    );
}

/// Oracle for binary search
pub trait OracleBS {
    fn assess_bs(&mut self, t: f64) -> bool;
}

pub trait SearchSpace {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    fn xc(&self) -> Self::ArrayType;
    fn update<T>(&mut self, cut: &(Self::ArrayType, T)) -> (CutStatus, f64)
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
 * @param ss    search Space containing x*
 * @param options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_feas<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    options: &Options,
) -> CInfo
where
    T: UpdateByCutChoices<Space, ArrayType = Oracle::ArrayType>,
    Oracle: OracleFeas<CutChoices = T>,
    Space: SearchSpace<ArrayType = Oracle::ArrayType>,
{
    for niter in 0..options.max_iter {
        let cut = omega.assess_feas(&space.xc()); // query the oracle at &space.xc()
        if cut.is_none() {
            // feasible sol'n obtained
            return (true, niter, CutStatus::Success);
        }
        let (cutstatus, tsq) = space.update::<T>(&cut.unwrap()); // update space
        if cutstatus != CutStatus::Success {
            return (false, niter, cutstatus);
        }
        if tsq < options.tol {
            return (false, niter, CutStatus::SmallEnough);
        }
    }
    (false, options.max_iter, CutStatus::NoSoln)
}

/**
 * @brief Cutting-plane method for solving convex problem
 *
 * @tparam Oracle
 * @tparam Space
 * @tparam opt_type
 * @param omega perform assessment on x0
 * @param ss    search Space containing x*
 * @param t     best-so-far optimal sol'n
 * @param options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_optim<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    t: &mut f64,
    options: &Options,
) -> (Option<Oracle::ArrayType>, usize, CutStatus)
where
    T: UpdateByCutChoices<Space, ArrayType = Oracle::ArrayType>,
    Oracle: OracleOptim<CutChoices = T>,
    Space: SearchSpace<ArrayType = Oracle::ArrayType>,
{
    let mut x_best: Option<Oracle::ArrayType> = None;
    let mut status = CutStatus::NoSoln;

    for niter in 0..options.max_iter {
        let (cut, shrunk) = omega.assess_optim(&space.xc(), t); // query the oracle at &space.xc()
        if shrunk {
            // best t obtained
            x_best = Some(space.xc());
            status = CutStatus::Success;
        }
        let (cutstatus, tsq) = space.update::<T>(&cut); // update ss
        if cutstatus != CutStatus::Success {
            return (x_best, niter, cutstatus);
        }
        if tsq < options.tol {
            return (x_best, niter, CutStatus::SmallEnough);
        }
    }
    (x_best, options.max_iter, status)
} // END

/**
    Cutting-plane method for solving convex discrete optimization problem
    input
             oracle        perform assessment on x0
             ss(xc)        Search space containing x*
             t             best-so-far optimal sol'n
             max_iter      maximum number of iterations
             tol           error tolerance
    output
             x             solution vector
             niter         number of iterations performed
**/

/**
 * @brief Cutting-plane method for solving convex discrete optimization problem
 *
 * @tparam Oracle
 * @tparam Space
 * @param omega perform assessment on x0
 * @param ss     search Space containing x*
 * @param t     best-so-far optimal sol'n
 * @param options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_q<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    t: &mut f64,
    options: &Options,
) -> (Option<Oracle::ArrayType>, usize, CutStatus)
where
    T: UpdateByCutChoices<Space, ArrayType = Oracle::ArrayType>,
    Oracle: OracleQ<CutChoices = T>,
    Space: SearchSpace<ArrayType = Oracle::ArrayType>,
{
    let mut x_best: Option<Oracle::ArrayType> = None;
    // let mut status = CutStatus::NoSoln; // note!!!
    let mut retry = false;

    for niter in 0..options.max_iter {
        let (cut, shrunk, x0, more_alt) = omega.assess_q(&space.xc(), t, retry); // query the oracle at &space.xc()
        if shrunk {
            // best t obtained
            x_best = Some(x0); // x0
        }
        let (status, tsq) = space.update::<T>(&cut); // update space
        match &status {
            CutStatus::NoEffect => {
                if !more_alt {
                    // more alt?
                    return (x_best, niter, status);
                }
                // status = cutstatus;
                retry = true;
            }
            CutStatus::NoSoln => {
                return (x_best, niter, CutStatus::NoSoln);
            }
            _ => {}
        }
        if tsq < options.tol {
            return (x_best, niter, CutStatus::SmallEnough);
        }
    }
    (x_best, options.max_iter, CutStatus::Success)
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
pub fn bsearch<Oracle>(omega: &mut Oracle, intvl: &mut (f64, f64), options: &Options) -> CInfo
where
    Oracle: OracleBS,
{
    // assume monotone
    // auto& [lower, upper] = I;
    let &mut (mut lower, mut upper) = intvl;
    assert!(lower <= upper);
    let u_orig = upper;

    for niter in 0..options.max_iter {
        let tau = (upper - lower) / 2.0;
        if tau < options.tol {
            return (upper != u_orig, niter, CutStatus::SmallEnough);
        }
        let mut t = lower; // l may be `i32` or `Fraction`
        t += tau;
        if omega.assess_bs(t) {
            // feasible sol'n obtained
            upper = t;
        } else {
            lower = t;
        }
    }
    (upper != u_orig, options.max_iter, CutStatus::Unknown)
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
//     Oracle& _P;
//     Space& _S;
//     const Options _options;

//   public:
//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param P perform assessment on x0
//      * @param ss search Space containing x*
//      */
//     bsearch_adaptor(Oracle& P, Space& ss) : bsearch_adaptor{P, ss, Options()} {}

//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param P perform assessment on x0
//      * @param ss search Space containing x*
//      * @param options maximum iteration and error tolerance etc.
//      */
//     bsearch_adaptor(Oracle& P, Space& ss, const Options& options)
//         : _P{P}, _S{ss}, _options{options} {}

//     /**
//      * @brief get best x
//      *
//      * @return auto
//      */
//     let mut x_best() const { return self.&ss.xc(); }

//     /**
//      * @brief
//      *
//      * @param t the best-so-far optimal value
//      * @return bool
//      */
//     template <typename opt_type> let mut operator()(const opt_type& t) -> bool {
//         Space ss = self.ss.copy();
//         self.P.update(t);
//         let ell_info = cutting_plane_feas(self.P, ss, self.options);
//         if ell_info.feasible {
//             self.ss.set_xc(&ss.xc());
//         }
//         return ell_info.feasible;
//     }
// };
