use ndarray::{Array1, ArrayView1, ArrayView2};
use ndarray_linalg::Norm;

pub struct Options {
    pub max_iters: usize,
    pub tolerance: f64,
}

pub fn power_iteration(a: ArrayView2<f64>, mut x: Array1<f64>, options: &Options) -> (Array1<f64>, f64, usize, f64) {
    x /= x.dot(&x).sqrt();
    for niter in 0..options.max_iters {
        let mut x1 = a.dot(&x);
        let x1_norm = x1.dot(&x1).sqrt();
        x1 /= x1_norm;
        let eps = (&x - &x1).norm_l1();
        if eps <= options.tolerance {
            return (x1.clone(), x1.dot(&a.dot(&x1)), niter, eps);
        }
        let eps = (&x + &x1).norm_l1();
        if eps <= options.tolerance {
            return (x1.clone(), x1.dot(&a.dot(&x1)), niter, eps);
        }
        x = x1;
    }
    let ld = x.dot(&a.dot(&x));
    (x, ld, options.max_iters, f64::INFINITY)
}

pub fn calc_core2(a: ArrayView2<f64>, mut x: Array1<f64>) -> (Array1<f64>, f64) {
    x /= x.dot(&x).sqrt();
    let new = a.dot(&x);
    let ld = x.dot(&new);
    (new, ld)
}

pub fn power_iteration2(a: ArrayView2<f64>, x: Array1<f64>, options: &Options) -> (Array1<f64>, f64, usize, f64) {
    let (mut new, mut ld) = calc_core2(a, x);
    for niter in 0..options.max_iters {
        let ld1 = ld;
        let x = new;
        let (new_temp, ld_temp) = calc_core2(a, x);
        new = new_temp;
        ld = ld_temp;
        let eps = (ld1 - ld).abs();
        if eps <= options.tolerance {
            return (new, ld, niter, eps);
        }
    }
    (new, ld, options.max_iters, f64::INFINITY)
}

pub fn calc_core3(a: ArrayView2<f64>, x: ArrayView1<f64>) -> (Array1<f64>, f64, f64) {
    let new = a.dot(&x);
    let dot = x.dot(&x);
    let ld = x.dot(&new) / dot;
    (new, dot, ld)
}

pub fn power_iteration3(a: ArrayView2<f64>, mut x: Array1<f64>, options: &Options) -> (Array1<f64>, f64, usize, f64) {
    let (mut new, mut dot, mut ld) = calc_core3(a, x.view());
    for niter in 0..options.max_iters {
        let ld1 = ld;
        x = new.clone();
        let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
        if xmax >= 1e100 || xmin <= -1e100 {
            x /= 1e100;
        }
        let (new_temp, dot_temp, ld_temp) = calc_core3(a, x.view());
        new = new_temp;
        dot = dot_temp;
        ld = ld_temp;
        let eps = (ld1 - ld).abs();
        if eps <= options.tolerance {
            x = new.clone() / dot.sqrt();
            return (x, ld, niter, eps);
        }
    }
    x = new / dot.sqrt();
    (x, ld, options.max_iters, f64::INFINITY)
}

pub fn power_iteration4(a: ArrayView2<f64>, mut x: Array1<f64>, options: &Options) -> (Array1<f64>, f64, usize, f64) {
    x /= x.iter().cloned().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
    for niter in 0..options.max_iters {
        let mut x1 = a.dot(&x);
        x1 /= x1.iter().cloned().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
        let eps = (&x - &x1).norm_l1();
        if eps <= options.tolerance {
            x1 /= x1.dot(&x1).sqrt();
            let ld = x1.dot(&a.dot(&x1));
            return (x1, ld, niter, eps);
        }
        let eps = (&x + &x1).norm_l1();
        if eps <= options.tolerance {
            x1 /= x1.dot(&x1).sqrt();
            let ld = x1.dot(&a.dot(&x1));
            return (x1, ld, niter, eps);
        }
        x = x1;
    }
    x /= x.dot(&x).sqrt();
    let ld = x.dot(&a.dot(&x));
    (x, ld, options.max_iters, f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_construct() {

        let a = Array2::from_shape_vec((2, 2), vec![3.7, -4.5, 4.3, -5.9]).unwrap();
        let x = Array1::from_vec(vec![1.1, 0.0]);
        let options = Options { max_iters: 2000, tolerance: 1e-9 };
    
        println!("1-----------------------------");
        let (x, ld, niter, eps) = power_iteration(a.view(), x.clone(), &options);
        println!("{:?}", x);
        println!("{}", ld);
        println!("{}", niter);
        println!("{}", eps);
    
        println!("2-----------------------------");
        let (x, ld, niter, eps) = power_iteration2(a.view(), x.clone(), &options);
        println!("{:?}", x);
        println!("{}", ld);
        println!("{}", niter);
        println!("{}", eps);
    
        println!("3-----------------------------");
        let (x, ld, niter, eps) = power_iteration3(a.view(), x.clone(), &options);
        println!("{:?}", x);
        println!("{}", ld);
        println!("{}", niter);
        println!("{}", eps);
    
        println!("4-----------------------------");
        let (x, ld, niter, eps) = power_iteration4(a.view(), x.clone(), &options);
        println!("{:?}", x);
        println!("{}", ld);
        println!("{}", niter);
        println!("{}", eps);
    }
}
