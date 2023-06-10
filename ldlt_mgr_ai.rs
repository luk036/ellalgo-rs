use ndarray::{Array, Array1, Array2};
use ndarray_linalg::Lapack;
use std::cmp::min;

struct LDLTMgr {
    p: (usize, usize),
    v: Array1<f64>,
    n: usize,
    t: Array2<f64>,
}

impl LDLTMgr {
    fn new(n: usize) -> Self {
        Self {
            p: (0, 0),
            v: Array1::zeros(n),
            n,
            t: Array2::zeros((n, n)),
        }
    }

    fn factorize(&mut self, a: &Array2<f64>) -> bool {
        self.factor(&|i, j| a[[i, j]])
    }

    fn factor<F>(&mut self, get_elem: &F) -> bool
    where
        F: Fn(usize, usize) -> f64,
    {
        let mut start = 0;
        self.p = (0, 0);
        for i in 0..self.n {
            let mut d = get_elem(i, start);
            for j in start..i {
                self.t[[j, i]] = d;
                self.t[[i, j]] = d / self.t[[j, j]];
                let s = j + 1;
                d = get_elem(i, s)
                    - self.t.slice(s..i, start..s).dot(&self.t.slice(s..i, s..).dot(&self.t.slice(start..s, s..)))
            }
            self.t[[i, i]] = d;
            if d <= 0.0 {
                self.p = (start, i + 1);
                break;
            }
        }
        self.is_spd()
    }

    fn is_spd(&self) -> bool {
        self.p.1 == 0
    }

    fn witness(&mut self) -> f64 {
        if self.is_spd() {
            panic!("Matrix is SPD");
        }
        let (start, n) = self.p;
        let m = n - 1;
        self.v[m] = 1.0;
        for i in (start..m).rev() {
            self.v.slice_mut(i..=m).assign(&(-self.t.slice(i + 1..=m, i).dot(&self.v.slice(i + 1..=m))));
        }
        -self.t[[m, m]]
    }

    fn sym_quad(&self, a: &Array2<f64>) -> f64 {
        let (s, n) = self.p;
        let v = self.v.slice(s..n);
        v.dot(&a.slice(s..n, s..n).dot(&v))
    }
}

fn main() {
    let n = 3;
    let mut ldlt = LDLTMgr::new(n);
    let a = Array::from_shape_vec((n, n), vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]).unwrap();
    ldlt.factorize(&a);
    assert!(ldlt.is_spd());
    let b = Array::from_shape_vec((n, n), vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 99.0]).unwrap();
    ldlt.factorize(&b);
    assert!(!ldlt.is_spd());
    assert_eq!(ldlt.witness(), -1.0);
    assert_eq!(ldlt.sym_quad(&a), 16.0);
}

