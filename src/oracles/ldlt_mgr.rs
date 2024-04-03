use ndarray::{s, Array1, Array2};
// use ndarray_linalg::Lapack;
// use std::cmp::min;

/// The `LDLTMgr` struct is a manager for LDLTMgr factorization in Rust.
///
/// `LDLTMgr` is a class that performs the LDLTMgr factorization for a given
/// symmetric matrix. The LDLTMgr factorization decomposes a symmetric matrix A into
/// the product of a lower triangular matrix L, a diagonal matrix D, and the
/// transpose of L. This factorization is useful for solving linear systems and
/// eigenvalue problems. The class provides methods to perform the factorization,
/// check if the matrix is positive definite, calculate a witness vector if it is
/// not positive definite, and calculate the symmetric quadratic form.
///
///  - LDL^T square-root-free version
///  - Option allow semidefinite
///  - A matrix A in R^{m x m} is positive definite iff v' A v > 0
///      for all v in R^n.
///  - O(p^2) per iteration, independent of N
///
/// Properties:
///
/// * `pos`: A tuple containing two usize values. It represents the dimensions of the LDLTMgr factorization.
/// * `wit`: The `wit` property is an Array1 of f64 values.
/// * `ndim`: The `ndim` property represents the size of the matrix that will be factorized using LDLTMgr
/// factorization.
/// * `storage`: The `storage` property is a 2-dimensional array of type `f64`. It is used to store the LDLTMgr
/// factorization of a matrix.
pub struct LDLTMgr {
    pub pos: (usize, usize),
    pub wit: Array1<f64>,
    pub ndim: usize,
    pub storage: Array2<f64>,
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
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::oracles::ldlt_mgr::LDLTMgr;
    /// use ndarray::{array, Array1};
    /// let ldlt_mgr = LDLTMgr::new(3);
    /// assert_eq!(ldlt_mgr.pos, (0, 0));
    /// assert_eq!(ldlt_mgr.wit.len(), 3);
    /// assert_eq!(ldlt_mgr.ndim, 3);
    /// assert_eq!(ldlt_mgr.storage.len(), 9);
    /// ```
    pub fn new(ndim: usize) -> Self {
        Self {
            pos: (0, 0),
            wit: Array1::zeros(ndim),
            ndim,
            storage: Array2::zeros((ndim, ndim)),
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
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::oracles::ldlt_mgr::LDLTMgr;
    /// use ndarray::array;
    /// let mut ldlt_mgr = LDLTMgr::new(3);
    /// let mat_a = array![
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 4.0, 5.0],
    ///     [3.0, 5.0, 6.0],
    /// ];
    /// assert_eq!(ldlt_mgr.factorize(&mat_a), false);
    /// ```
    pub fn factorize(&mut self, mat_a: &Array2<f64>) -> bool {
        self.factor(&|i, j| mat_a[[i, j]])
    }

    /// The `factor` function performs LDLTMgr factorization on a matrix and checks if it is symmetric
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
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::oracles::ldlt_mgr::LDLTMgr;
    /// use ndarray::array;
    /// let mut ldlt_mgr = LDLTMgr::new(3);
    /// let mat_a = array![
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 4.0, 5.0],
    ///     [3.0, 5.0, 6.0],
    /// ];
    /// assert_eq!(ldlt_mgr.factor(&|i, j| mat_a[[i, j]]), false);
    /// ```
    pub fn factor<F>(&mut self, get_elem: &F) -> bool
    where
        F: Fn(usize, usize) -> f64,
    {
        self.pos = (0, 0);
        for i in 0..self.ndim {
            let mut diag = get_elem(i, 0);
            for j in 0..i {
                self.storage[[j, i]] = diag;
                self.storage[[i, j]] = diag / self.storage[[j, j]];
                let stop = j + 1;
                // diag = get_elem(i, stop);
                // for k in 0..stop {
                //     diag -= self.storage[[i, k]] * self.storage[[k, stop]];
                // }
                diag = get_elem(i, stop)
                    - self
                        .storage
                        .slice(s![i, 0..stop])
                        .dot(&self.storage.slice(s![0..stop, stop]));
            }
            self.storage[[i, i]] = diag;
            if diag <= 0.0 {
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
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::oracles::ldlt_mgr::LDLTMgr;
    /// use ndarray::array;
    /// let mut ldlt_mgr = LDLTMgr::new(3);
    /// let mat_a = array![
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 4.0, 5.0],
    ///     [3.0, 5.0, 6.0],
    /// ];
    /// assert_eq!(ldlt_mgr.factor_with_allow_semidefinite(&|i, j| mat_a[[i, j]]), true);
    /// ```
    pub fn factor_with_allow_semidefinite<F>(&mut self, get_elem: &F) -> bool
    where
        F: Fn(usize, usize) -> f64,
    {
        self.pos = (0, 0);
        let mut start = 0;
        for i in 0..self.ndim {
            let mut diag = get_elem(i, start);
            for j in start..i {
                self.storage[[j, i]] = diag;
                self.storage[[i, j]] = diag / self.storage[[j, j]];
                let stop = j + 1;
                // diag = get_elem(i, stop);
                // for k in start..stop {
                //     diag -= self.storage[[i, k]] * self.storage[[k, stop]];
                // }
                diag = get_elem(i, stop)
                    - self
                        .storage
                        .slice(s![i, start..stop])
                        .dot(&self.storage.slice(s![start..stop, stop]));
            }
            self.storage[[i, i]] = diag;
            if diag < 0.0 {
                self.pos = (start, i + 1);
                break;
            }
            if diag == 0.0 {
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
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::oracles::ldlt_mgr::LDLTMgr;
    /// use ndarray::array;
    /// let mut ldlt_mgr = LDLTMgr::new(3);
    /// let mat_a = array![
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 4.0, 5.0],
    ///     [3.0, 5.0, 6.0],
    /// ];
    /// assert_eq!(ldlt_mgr.factor(&|i, j| mat_a[[i, j]]), false);
    /// assert_eq!(ldlt_mgr.is_spd(), false);
    /// ```
    pub fn is_spd(&self) -> bool {
        self.pos.1 == 0
    }

    /// The function calculates the witness vector of a matrix.
    ///
    /// Returns:
    ///
    /// The function `witness` returns a `f64` (a 64-bit floating-point number).
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::oracles::ldlt_mgr::LDLTMgr;
    /// use ndarray::array;
    /// let mut ldlt_mgr = LDLTMgr::new(3);
    /// let mat_a = array![
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 4.0, 5.0],
    ///     [3.0, 5.0, 6.0],
    /// ];
    /// assert_eq!(ldlt_mgr.factor(&|i, j| mat_a[[i, j]]), false);
    /// assert_eq!(ldlt_mgr.witness(), 0.0);
    /// assert_eq!(ldlt_mgr.wit[0], -2.0);
    /// assert_eq!(ldlt_mgr.wit[1], 1.0);
    /// assert_eq!(ldlt_mgr.pos.1, 2);
    /// ```
    pub fn witness(&mut self) -> f64 {
        if self.is_spd() {
            panic!("Matrix is symmetric positive definite");
        }
        let (start, ndim) = self.pos;
        let m = ndim - 1;
        self.wit[m] = 1.0;
        for i in (start..m).rev() {
            self.wit[i] = 0.0;
            for k in i..ndim {
                self.wit[i] -= self.storage[[k, i]] * self.wit[k];
            }
        }
        -self.storage[[m, m]]
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
    ///
    /// # Examples
    ///
    /// ```
    /// use ellalgo_rs::oracles::ldlt_mgr::LDLTMgr;
    /// use ndarray::array;
    /// let mut ldlt_mgr = LDLTMgr::new(3);
    /// let mat_a = array![
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 4.0, 5.0],
    ///     [3.0, 5.0, 6.0],
    /// ];
    /// assert_eq!(ldlt_mgr.factor(&|i, j| mat_a[[i, j]]), false);
    /// assert_eq!(ldlt_mgr.witness(), 0.0);
    /// let mat_b = array![
    ///     [1.0, 2.0, 3.0],
    ///     [2.0, 6.0, 5.0],
    ///     [3.0, 5.0, 4.0],
    /// ];
    /// assert_eq!(ldlt_mgr.sym_quad(&mat_b), 2.0);
    pub fn sym_quad(&self, mat_a: &Array2<f64>) -> f64 {
        let mut res = 0.0;
        let (start, stop) = self.pos;
        for i in start..stop {
            let mut s = 0.0;
            for j in (i + 1)..stop {
                s += mat_a[[i, j]] * self.wit[j];
            }
            res += self.wit[i] * (mat_a[[i, i]] * self.wit[i] + 2.0 * s);
        }
        res
    }

    pub fn sqrt(&self) -> Array2<f64> {
        if !self.is_spd() {
            panic!("Matrix is not symmetric positive definite");
        }
        let mut res = Array2::zeros((self.ndim, self.ndim));
        for i in 0..self.ndim {
            res[[i, i]] = self.storage[[i, i]].sqrt();
            for j in (i + 1)..self.ndim {
                res[[i, j]] = self.storage[[j, i]] * res[[i, i]];
            }
        }
        res
    }
}

// pub fn test_ldlt() {
//     let ndim = 3;
//     let mut ldlt_mgr = LDLTMgr::new(ndim);
//     let a = Array::from_shape_vec((ndim, ndim), vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]).unwrap();
//     ldlt_mgr.factorize(&a);
//     assert!(ldlt_mgr.is_spd());
//     let b = Array::from_shape_vec((ndim, ndim), vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 99.0]).unwrap();
//     ldlt_mgr.factorize(&b);
//     assert!(!ldlt_mgr.is_spd());
//     assert_eq!(ldlt_mgr.witness(), -1.0);
//     assert_eq!(ldlt_mgr.sym_quad(&a), 16.0);
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ShapeError};

    #[test]
    fn test_chol1() -> Result<(), ShapeError> {
        let l1 = Array2::from_shape_vec(
            (3, 3),
            vec![25.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        assert!(ldlt_mgr.factorize(&l1));
        Ok(())
    }

    #[test]
    fn test_chol2() -> Result<(), ShapeError> {
        let l2 = Array2::from_shape_vec(
            (4, 4),
            vec![
                18.0, 22.0, 54.0, 42.0, 22.0, -70.0, 86.0, 62.0, 54.0, 86.0, -174.0, 134.0, 42.0,
                62.0, 134.0, -106.0,
            ],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(4);
        assert!(!ldlt_mgr.factorize(&l2));
        ldlt_mgr.witness();
        assert_eq!(ldlt_mgr.pos, (0, 2));
        Ok(())
    }

    #[test]
    fn test_chol3() -> Result<(), ShapeError> {
        let l3 = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        assert!(!ldlt_mgr.factorize(&l3));
        let ep = ldlt_mgr.witness();
        assert_eq!(ldlt_mgr.pos, (0, 1));
        assert_eq!(ldlt_mgr.wit[0], 1.0);
        assert_eq!(ep, 0.0);
        Ok(())
    }

    #[test]
    fn test_chol4() -> Result<(), ShapeError> {
        let l1 = Array2::from_shape_vec(
            (3, 3),
            vec![25.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        assert!(ldlt_mgr.factor_with_allow_semidefinite(&|i, j| l1[[i, j]]));
        Ok(())
    }

    #[test]
    fn test_chol5() -> Result<(), ShapeError> {
        let l2 = Array2::from_shape_vec(
            (4, 4),
            vec![
                18.0, 22.0, 54.0, 42.0, 22.0, -70.0, 86.0, 62.0, 54.0, 86.0, -174.0, 134.0, 42.0,
                62.0, 134.0, -106.0,
            ],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(4);
        assert!(!ldlt_mgr.factor_with_allow_semidefinite(&|i, j| l2[[i, j]]));
        ldlt_mgr.witness();
        assert_eq!(ldlt_mgr.pos, (0, 2));
        Ok(())
    }

    #[test]
    fn test_chol6() -> Result<(), ShapeError> {
        let l3 = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        assert!(ldlt_mgr.factor_with_allow_semidefinite(&|i, j| l3[[i, j]]));
        Ok(())
    }

    #[test]
    fn test_chol7() -> Result<(), ShapeError> {
        let l3 = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, -20.0],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        assert!(!ldlt_mgr.factor_with_allow_semidefinite(&|i, j| l3[[i, j]]));
        let ep = ldlt_mgr.witness();
        assert_eq!(ep, 20.0);
        Ok(())
    }

    #[test]
    fn test_chol8() -> Result<(), ShapeError> {
        let l3 = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 20.0],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        assert!(!ldlt_mgr.factorize(&l3));
        Ok(())
    }

    #[test]
    fn test_chol9() -> Result<(), ShapeError> {
        let l3 = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 20.0],
        )?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        assert!(ldlt_mgr.factor_with_allow_semidefinite(&|i, j| l3[[i, j]]));
        Ok(())
    }

    #[test]
    fn test_ldlt_mgr_sqrt() -> Result<(), ShapeError> {
        let a =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.5, 0.5, 0.5, 1.25, 0.75, 0.5, 0.75, 1.5])?;
        let mut ldlt_mgr = LDLTMgr::new(3);
        ldlt_mgr.factorize(&a);
        assert!(ldlt_mgr.is_spd());
        let r = ldlt_mgr.sqrt();
        assert_eq!(
            r,
            Array2::from_shape_vec((3, 3), vec![1.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0])?
        );
        Ok(())
    }
}
