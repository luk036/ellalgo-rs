pub type Arr = [f64; 100];
pub type Cut = (Arr, f64);
pub type CInfo = (bool, u32, CutStatus);

pub struct Options {
    pub max_it: u32,
    pub tol: f64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CutStatus {
    Success,
    NoSoln,
    NoEffect,
    SmallEnough,
}

pub trait OracleFeas {
    fn asset_feas(&mut self, x: &Arr) -> Option<Cut>;
}

pub trait OracleOptim {
    fn asset_optim(&mut self, x: &Arr, t: &mut f64) -> (Cut, bool);
}

pub trait OracleQ {
    fn asset_q(&mut self, x: &Arr, t: &mut f64, retry: bool) -> (Cut, bool, Arr, bool);
}

pub trait OracleBS {
    fn asset_bs(&mut self, t: f64) -> bool;
}

pub trait SearchSpace {
    fn xc(&self) -> Arr;
    fn update(&mut self, cut: &(Arr, f64)) -> (CutStatus, f64);
}

/**
 * @brief Find a point in a convex set (defined through a cutting-plane oracle).
 *
 *     A function f(x) is *convex* if there always exist a g(x)
 *     such that f(z) >= f(x) + g(x)' * (z - x), forall z, x in dom f.
 *     Note that dom f does not need to be a convex set in our definition.
 *     The affine function g' (x - xc) + beta is called a cutting-plane,
 *     or a ``cut'' for short.
 *     This algorithm solves the following feasibility problem:
 *
 *             find x
 *             s.t. f(x) <= 0,
 *
 *     A *separation oracle* asserts that an evalution point x0 is feasible,
 *     or provide a cut that separates the feasible region and x0.
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss     search Space containing x*
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_feas<Oracle, Space>(
    omega: &mut Oracle,
    ss: &mut Space,
    options: &Options,
) -> CInfo
where
    Oracle: OracleFeas,
    Space: SearchSpace,
{
    let mut feasible = false;
    let mut status = CutStatus::NoSoln;

    let mut niter = 0;
    while niter < options.max_it {
        niter += 1;
        let cut = omega.asset_feas(&ss.xc()); // query the oracle at &ss.xc()
        if let Some(value) = cut {
            // feasible sol'n obtained
            let (cutstatus, tsq) = ss.update(&value); // update ss
            if cutstatus != CutStatus::Success {
                status = cutstatus;
                break;
            }
            if tsq < options.tol {
                // no more
                status = CutStatus::SmallEnough;
                break;
            }
        } else {
            feasible = true;
            status = CutStatus::Success;
            break;
        }
    }
    (feasible, niter, status)
}

/**
 * @brief Cutting-plane method for solving convex problem
 *
 * @tparam Oracle
 * @tparam Space
 * @tparam opt_type
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss     search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_optim<Oracle, Space>(
    omega: &mut Oracle,
    ss: &mut Space,
    t: &mut f64,
    options: &Options,
) -> (Option<Arr>, u32, CutStatus)
where
    Oracle: OracleOptim,
    Space: SearchSpace,
{
    let mut x_best: Option<Arr> = None;
    let mut status = CutStatus::NoSoln;
    let t_orig = *t;

    let mut niter = 0;
    while niter < options.max_it {
        niter += 1;
        let (cut, shrunk) = omega.asset_optim(&ss.xc(), t); // query the oracle at &ss.xc()
        if shrunk {
            // best t obtained
            x_best = Some(ss.xc());
            status = CutStatus::Success;
        }
        let (cutstatus, tsq) = ss.update(&cut); // update ss
        if cutstatus != CutStatus::Success {
            status = cutstatus;
            break;
        }
        if tsq < options.tol {
            // no more
            status = CutStatus::SmallEnough;
            break;
        }
    }
    (x_best, niter, status)
} // END

/**
    Cutting-plane method for solving convex discrete optimization problem
    input
             oracle        perform assessment on x0
             ss(xc)         Search space containing x*
             t             best-so-far optimal sol'n
             max_it        maximum number of iterations
             tol           error tolerance
    output
             x             solution vector
             niter          number of iterations performed
**/
// #include <boost/numeric/ublas/symmetric.hpp>
// namespace bnu = boost::numeric::ublas;
// #include <xtensor-blas/xlinalg.hpp>
// #include <xtensor/xarray.hpp>

/**
 * @brief Cutting-plane method for solving convex discrete optimization problem
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss     search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
pub fn cutting_plane_q<Oracle, Space>(
    omega: &mut Oracle,
    ss: &mut Space,
    t: &mut f64,
    options: &Options,
) -> (Option<Arr>, u32, CutStatus)
where
    Oracle: OracleQ,
    Space: SearchSpace,
{
    let mut x_best: Option<Arr> = None;
    let mut status = CutStatus::NoSoln; // note!!!
    let mut retry = false;

    let mut niter = 0;
    while niter < options.max_it {
        niter += 1;
        let (cut, shrunk, x0, more_alt) = omega.asset_q(&ss.xc(), t, retry); // query the oracle at &ss.xc()
        if shrunk {
            // best t obtained
            x_best = Some(x0); // x0
        }
        let (cutstatus, tsq) = ss.update(&cut); // update ss
        match &cutstatus {
            CutStatus::NoEffect => {
                if !more_alt {
                    // more alt?
                    break; // no more alternative cut
                }
                status = cutstatus;
                retry = true;
            }
            CutStatus::NoSoln => {
                status = cutstatus;
                break;
            }
            _ => {}
        }
        if tsq < options.tol {
            status = CutStatus::SmallEnough;
            break;
        }
    }
    (x_best, niter, status)
} // END

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega    perform assessment on x0
 * @param[in,out] I        interval containing x*
 * @param[in]     options  maximum iteration and error tolerance etc.
 * @return CInfo
 */
#[allow(dead_code)]
pub fn besearch<Oracle>(omega: &mut Oracle, intvl: &mut (f64, f64), options: &Options) -> CInfo
where
    Oracle: OracleBS,
{
    // assume monotone
    // auto& [lower, upper] = I;
    let &mut (mut lower, mut upper) = intvl;
    assert!(lower <= upper);
    let u_orig = upper;
    let mut status = CutStatus::NoSoln;

    let mut niter = 0;
    while niter < options.max_it {
        niter += 1;
        let tau = (upper - lower) / 2.0;
        if tau < options.tol {
            status = CutStatus::SmallEnough;
            break;
        }
        let mut t = lower; // l may be `i32` or `Fraction`
        t += tau;
        if omega.asset_bs(t) {
            // feasible sol'n obtained
            upper = t;
        } else {
            lower = t;
        }
    }
    (upper != u_orig, niter, status)
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
//      * @param[in,out] P perform assessment on x0
//      * @param[in,out] ss search Space containing x*
//      */
//     bsearch_adaptor(Oracle& P, Space& ss) : bsearch_adaptor{P, ss, Options()} {}

//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param[in,out] P perform assessment on x0
//      * @param[in,out] ss search Space containing x*
//      * @param[in] options maximum iteration and error tolerance etc.
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
//      * @param[in,out] t the best-so-far optimal value
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
