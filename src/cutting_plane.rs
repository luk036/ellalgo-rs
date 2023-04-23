#![allow(non_snake_case)]

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

type CInfo = (bool, usize);

/// TODO: support 1D problems

pub trait UpdateByCutChoices<SS> {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    fn update_by(&self, space: &mut SS, grad: &Self::ArrayType) -> CutStatus;
}

/// Oracle for feasibility problems
pub trait OracleFeas<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_feas(&mut self, x: &ArrayType) -> Option<(ArrayType, Self::CutChoices)>;
}

/// Oracle for optimization problems
pub trait OracleOptim<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64>) for parallel cut
    fn assess_optim(
        &mut self,
        x: &ArrayType,
        tea: &mut f64,
    ) -> ((ArrayType, Self::CutChoices), bool);
}

/// Oracle for quantized optimization problems
pub trait OracleQ<ArrayType> {
    type CutChoices; // f64 for single cut; (f64, Option<f64) for parallel cut
    fn assess_q(
        &mut self,
        x: &ArrayType,
        tea: &mut f64,
        retry: bool,
    ) -> (
        (ArrayType, Self::CutChoices),
        Option<ArrayType>,
        bool,
    );
}

/// Oracle for binary search
pub trait OracleBS {
    fn assess_bs(&mut self, tea: f64) -> bool;
}

pub trait SearchSpace {
    type ArrayType; // f64 for 1D; ndarray::Array1<f64> for general
    fn xc(&self) -> Self::ArrayType;
    fn tsq(&self) -> f64;
    fn update<T>(&mut self, cut: &(Self::ArrayType, T)) -> CutStatus
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
    for niter in 0..options.max_iter {
        let cut = omega.assess_feas(&space.xc()); // query the oracle at &space.xc()
        if cut.is_none() {
            // feasible sol'n obtained
            return (true, niter);
        }
        let cutstatus = space.update::<T>(&cut.unwrap()); // update space
        if cutstatus != CutStatus::Success || space.tsq() < options.tol {
            return (false, niter);
        }
    }
    (false, options.max_iter)
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

    for niter in 0..options.max_iter {
        let (cut, shrunk) = omega.assess_optim(&space.xc(), tea); // query the oracle at &space.xc()
        if shrunk {
            // best tea obtained
            x_best = Some(space.xc());
        }
        let cutstatus = space.update::<T>(&cut); // update space
        if cutstatus != CutStatus::Success || space.tsq() < options.tol {
            return (x_best, niter);
        }
    }
    (x_best, options.max_iter)
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
pub fn cutting_plane_q<T, Oracle, Space>(
    omega: &mut Oracle,
    space: &mut Space,
    tea: &mut f64,
    options: &Options,
) -> (Option<Space::ArrayType>, usize)
where
    T: UpdateByCutChoices<Space, ArrayType = Space::ArrayType>,
    Oracle: OracleQ<Space::ArrayType, CutChoices = T>,
    Space: SearchSpace,
{
    let mut x_best: Option<Space::ArrayType> = None;
    // let mut status = CutStatus::NoSoln; // note!!!
    let mut retry = false;

    for niter in 0..options.max_iter {
        let (cut, x_opt, more_alt) = omega.assess_q(&space.xc(), tea, retry); // query the oracle at &space.xc()
        if let Some(x0) = x_opt {
            // best tea obtained
            x_best = Some(x0); // x0
        }
        let status = space.update::<T>(&cut); // update space
        match &status {
            CutStatus::NoEffect => {
                if !more_alt {
                    // more alt?
                    return (x_best, niter);
                }
                // status = cutstatus;
                retry = true;
            }
            CutStatus::NoSoln => {
                return (x_best, niter);
            }
            _ => {}
        }
        if space.tsq() < options.tol {
            return (x_best, niter);
        }
    }
    (x_best, options.max_iter)
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
    (upper != u_orig, options.max_iter)
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
