use ndarray::{Array1, ArrayView2};
// use ndarray_linalg::Norm;

pub struct Options {
    pub max_iters: usize,
    pub tolerance: f64,
}

#[inline]
fn norm_l1(x: &Array1<f64>) -> f64 {
    x.iter().cloned().map(|x| x.abs()).sum()
}

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
