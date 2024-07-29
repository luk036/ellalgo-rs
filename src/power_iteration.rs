use ndarray::{Array1, ArrayView2};
// use ndarray_linalg::Norm;

/// Configurable options for the power iteration algorithm.
///
/// The `max_iters` field specifies the maximum number of iterations to perform.
/// The `tolerance` field specifies the tolerance for convergence of the algorithm.
pub struct Options {
    pub max_iters: usize,
    pub tolerance: f64,
}

/// Computes the L1 norm (sum of absolute values) of the elements in the given array.
///
/// This function is an inline helper function used internally by the power iteration algorithm.
/// It calculates the L1 norm of the input array `x` by iterating through the elements and
/// summing their absolute values.
#[inline]
fn norm_l1(x: &Array1<f64>) -> f64 {
    x.iter().cloned().map(|x| x.abs()).sum()
}

/// Computes the dominant eigenvector of the given matrix `a` using the power iteration algorithm.
///
/// The power iteration algorithm is an iterative method for finding the dominant eigenvector of a matrix.
/// This function takes the matrix `a`, an initial guess for the eigenvector `x`, and a set of options
/// controlling the algorithm's behavior.
///
/// The function returns the dominant eigenvalue and the number of iterations performed.
pub fn power_iteration(a: ArrayView2<f64>, x: &mut Array1<f64>, options: &Options) -> (f64, usize) {
    *x /= x.dot(x).sqrt();
    for niter in 0..options.max_iters {
        let x1 = x.clone();
        *x = a.dot(&x1);
        *x /= x.dot(x).sqrt();
        if norm_l1(&(&*x - &x1)) <= options.tolerance || norm_l1(&(&*x + &x1)) <= options.tolerance
        {
            return (x.dot(&a.dot(x)), niter);
        }
    }
    (x.dot(&a.dot(x)), options.max_iters)
}

/// Computes the dominant eigenvector of the given matrix `a` using a modified power iteration algorithm.
///
/// This function is similar to the `power_iteration` function, but uses a different normalization
/// approach. Instead of normalizing the eigenvector by its L2 norm, it is normalized by its L1 norm.
/// This can sometimes lead to faster convergence, especially for sparse matrices.
///
/// The function takes the matrix `a`, an initial guess for the eigenvector `x`, and a set of options
/// controlling the algorithm's behavior. It returns the dominant eigenvalue and the number of
/// iterations performed.
pub fn power_iteration4(
    a: ArrayView2<f64>,
    x: &mut Array1<f64>,
    options: &Options,
) -> (f64, usize) {
    *x /= norm_l1(x);
    for niter in 0..options.max_iters {
        let x1 = x.clone();
        *x = a.dot(&x1);
        *x /= norm_l1(x);
        if norm_l1(&(&*x - &x1)) <= options.tolerance || norm_l1(&(&*x + &x1)) <= options.tolerance
        {
            *x /= x.dot(x).sqrt();
            return (x.dot(&a.dot(x)), niter);
        }
    }
    *x /= x.dot(x).sqrt();
    (x.dot(&a.dot(x)), options.max_iters)
}

/// Computes the dominant eigenvector of the given matrix `a` using a modified power iteration algorithm.
///
/// This function is similar to the `power_iteration` function, but uses a different normalization
/// approach. Instead of normalizing the eigenvector by its L2 norm, it is normalized by its L1 norm.
/// This can sometimes lead to faster convergence, especially for sparse matrices.
///
/// The function takes the matrix `a`, an initial guess for the eigenvector `x`, and a set of options
/// controlling the algorithm's behavior. It returns the dominant eigenvalue and the number of
/// iterations performed.
pub fn power_iteration2(
    a: ArrayView2<f64>,
    x: &mut Array1<f64>,
    options: &Options,
) -> (f64, usize) {
    // let (mut new, mut ld) = calc_core2(a, &mut x);
    *x /= x.dot(x).sqrt();
    let mut new = a.dot(x);
    let mut ld = x.dot(&new);
    for niter in 0..options.max_iters {
        let ld1 = ld;
        x.clone_from(&new);
        // let (new_temp, ld_temp) = calc_core2(a, &mut x);
        *x /= x.dot(x).sqrt();
        new = a.dot(x);
        ld = x.dot(&new);
        if (ld1 - ld).abs() <= options.tolerance {
            return (ld, niter);
        }
    }
    (ld, options.max_iters)
}

/// Computes the dominant eigenvector of the given matrix `a` using a modified power iteration algorithm.
///
/// This function is similar to the `power_iteration` and `power_iteration2` functions, but uses a different normalization
/// approach. It normalizes the eigenvector by its L2 norm, and also checks for very large values in the eigenvector
/// to prevent numerical overflow.
///
/// The function takes the matrix `a`, an initial guess for the eigenvector `x`, and a set of options
/// controlling the algorithm's behavior. It returns the dominant eigenvalue and the number of
/// iterations performed.
pub fn power_iteration3(
    a: ArrayView2<f64>,
    x: &mut Array1<f64>,
    options: &Options,
) -> (f64, usize) {
    let mut new = a.dot(x);
    let mut dot = x.dot(x);
    let mut ld = x.dot(&new) / dot;
    for niter in 0..options.max_iters {
        let ld1 = ld;
        x.clone_from(&new);
        dot = x.dot(x);
        if dot >= 1e150 {
            *x /= x.dot(x).sqrt();
            new = a.dot(x);
            ld = x.dot(&new);
            if (ld1 - ld).abs() <= options.tolerance {
                return (ld, niter);
            }
        } else {
            new = a.dot(x);
            ld = x.dot(&new) / dot;
            if (ld1 - ld).abs() <= options.tolerance {
                *x /= x.dot(x).sqrt();
                return (ld, niter);
            }
        }
    }
    (ld, options.max_iters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_construct() {
        let a = Array2::from_shape_vec(
            (3, 3),
            vec![3.7, -3.6, 0.7, -3.6, 4.3, -2.8, 0.7, -2.8, 5.4],
        )
        .unwrap();
        let options = Options {
            max_iters: 2000,
            tolerance: 1e-7,
        };

        println!("1-----------------------------");
        let mut x1 = Array1::from_vec(vec![0.3, 0.5, 0.4]);
        let (ld, niter) = power_iteration(a.view(), &mut x1, &options);
        println!("{:?}", x1);
        println!("{}", ld);
        assert_eq!(niter, 22);

        println!("4-----------------------------");
        let mut x4 = Array1::from_vec(vec![0.3, 0.5, 0.4]);
        let (ld, niter) = power_iteration4(a.view(), &mut x4, &options);
        println!("{:?}", x4);
        println!("{}", ld);
        assert_eq!(niter, 21);

        let options = Options {
            max_iters: 2000,
            tolerance: 1e-14,
        };

        println!("2-----------------------------");
        let mut x2 = Array1::from_vec(vec![0.3, 0.5, 0.4]);
        let (ld, niter) = power_iteration2(a.view(), &mut x2, &options);
        println!("{:?}", x2);
        println!("{}", ld);
        assert_eq!(niter, 23);

        println!("3-----------------------------");
        let mut x3 = Array1::from_vec(vec![0.3, 0.5, 0.4]);
        let (ld, niter) = power_iteration3(a.view(), &mut x3, &options);
        println!("{:?}", x3);
        println!("{}", ld);
        assert_eq!(niter, 23);
    }
}
