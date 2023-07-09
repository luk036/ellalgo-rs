use ndarray::{Array1, Array2};
// use ndarray_linalg::Lapack;
// use std::cmp::min;

/// The `LDLTMgr` struct is a manager for LDLT factorization in Rust.
///
/// Properties:
///
/// * `pos`: A tuple containing two usize values. It represents the dimensions of the LDLT factorization.
/// * `witness`: The `witness` property is an Array1 of f64 values.
/// * `ndim`: The `ndim` property represents the size of the matrix that will be factorized using LDLT
/// factorization.
/// * `mat_t`: The `mat_t` property is a 2-dimensional array of type `f64`. It is used to store the LDLT
/// factorization of a matrix.
pub struct LDLTMgr {
    pos: (usize, usize),
    witness: Array1<f64>,
    ndim: usize,
    mat_t: Array2<f64>,
}

impl LDLTMgr {
    /// The function `new` initializes a struct with default values for its fields.
    ///
    /// Arguments:
    ///
    /// * `ndim`: The parameter `ndim` represents the size of the arrays and matrices being initialized in the
    /// `new` function. It is of type `usize`, which means it represents a non-negative integer.
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the struct that it is defined in.
    pub fn new(ndim: usize) -> Self {
        Self {
            pos: (0, 0),
            witness: Array1::zeros(ndim),
            ndim,
            mat_t: Array2::zeros((ndim, ndim)),
        }
    }

    /// The `factorize` function takes a 2D array of f64 values and factors it using a closure.
    ///
    /// Arguments:
    ///
    /// * `mat_a`: The parameter `mat_a` is a reference to a 2-dimensional array (`Array2`) of `f64` values.
    ///
    /// Returns:
    ///
    /// The `factorize` function returns a boolean value.
    pub fn factorize(&mut self, mat_a: &Array2<f64>) -> bool {
        self.factor(&|i, j| mat_a[[i, j]])
    }

    /// The `factor` function performs LDLT factorization on a matrix and checks if it is symmetric
    /// positive definite.
    ///
    /// Arguments:
    ///
    /// * `get_elem`: `get_elem` is a closure that takes two `usize` parameters (`i` and `j`) and returns a
    /// `f64` value. It is used to retrieve elements from a matrix-like data structure.
    ///
    /// Returns:
    ///
    /// The `factor` function returns a boolean value.
    pub fn factor<F>(&mut self, get_elem: &F) -> bool
    where
        F: Fn(usize, usize) -> f64,
    {
        self.pos = (0, 0);
        for i in 0..self.ndim {
            let mut d = get_elem(i, 0);
            for j in 0..i {
                self.mat_t[[j, i]] = d;
                self.mat_t[[i, j]] = d / self.mat_t[[j, j]];
                let s = j + 1;
                d = get_elem(i, s);
                for k in 0..s {
                    d -= self.mat_t[[i, k]] * self.mat_t[[k, s]];
                }
            }
            self.mat_t[[i, i]] = d;
            if d <= 0.0 {
                self.pos = (0, i + 1);
                break;
            }
        }
        self.is_spd()
    }

    /// The function `factor_with_allow_semidefinite` checks if a given matrix is symmetric positive
    /// definite (SPD) or semidefinite.
    ///
    /// Arguments:
    ///
    /// * `get_elem`: `get_elem` is a closure that takes two `usize` parameters `i` and `start` and returns
    /// a `f64` value.
    ///
    /// Returns:
    ///
    /// a boolean value.
    pub fn factor_with_allow_semidefinite<F>(&mut self, get_elem: &F) -> bool
    where
        F: Fn(usize, usize) -> f64,
    {
        self.pos = (0, 0);
        let mut start = 0;
        for i in 0..self.ndim {
            let mut d = get_elem(i, start);
            for j in start..i {
                self.mat_t[[j, i]] = d;
                self.mat_t[[i, j]] = d / self.mat_t[[j, j]];
                let s = j + 1;
                d = get_elem(i, s);
                for k in start..s {
                    d -= self.mat_t[[i, k]] * self.mat_t[[k, s]];
                }
            }
            self.mat_t[[i, i]] = d;
            if d < 0.0 {
                self.pos = (start, i + 1);
                break;
            }
            if d == 0.0 {
                start = i + 1;
                // restart at i + 1, special as an LMI oracle
            }
        }
        self.is_spd()
    }

    /// The function `is_spd` checks if the second element of the tuple `pos` is equal to 0.
    ///
    /// Returns:
    ///
    /// A boolean value is being returned.
    pub fn is_spd(&self) -> bool {
        self.pos.1 == 0
    }

    /// The function calculates the witness vector of a matrix.
    ///
    /// Returns:
    ///
    /// The function `witness` returns a `f64` (a 64-bit floating-point number).
    pub fn witness(&mut self) -> f64 {
        if self.is_spd() {
            panic!("Matrix is SPD");
        }
        let (start, ndim) = self.pos;
        let m = ndim - 1;
        self.witness[m] = 1.0;
        for i in (start..m).rev() {
            self.witness[i] = 0.0;
            for k in i..ndim {
                self.witness[i] -= self.mat_t[[k, i]] * self.witness[k]
            }
        }
        -self.mat_t[[m, m]]
    }

    /// The `sym_quad` function calculates the quadratic form of a symmetric matrix and a vector.
    ///
    /// Arguments:
    ///
    /// * `mat_a`: A 2-dimensional array of type f64.
    ///
    /// Returns:
    ///
    /// The function `sym_quad` returns a `f64` value.
    pub fn sym_quad(&self, mat_a: &Array2<f64>) -> f64 {
        let mut res = 0.0;
        let (start, stop) = self.pos;
        for i in start..stop {
            let mut s = 0.0;
            for j in (i + 1)..stop {
                s += mat_a[[i, j]] * self.witness[j];
            }
            res += self.witness[i] * (mat_a[[i, i]] * self.witness[i] + 2.0 * s);
        }
        res
    }
}

// pub fn test_ldlt() {
//     let ndim = 3;
//     let mut ldlt = LDLTMgr::new(ndim);
//     let a = Array::from_shape_vec((ndim, ndim), vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]).unwrap();
//     ldlt.factorize(&a);
//     assert!(ldlt.is_spd());
//     let b = Array::from_shape_vec((ndim, ndim), vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 99.0]).unwrap();
//     ldlt.factorize(&b);
//     assert!(!ldlt.is_spd());
//     assert_eq!(ldlt.witness(), -1.0);
//     assert_eq!(ldlt.sym_quad(&a), 16.0);
// }
