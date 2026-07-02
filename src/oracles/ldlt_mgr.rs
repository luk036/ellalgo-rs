use crate::arr::Arr;

pub struct LDLTMgr {
    pub pos: (usize, usize),
    pub wit: Arr,
    ndim: usize,
    storage: Vec<f64>,
}

impl LDLTMgr {
    pub fn new(ndim: usize) -> Self {
        LDLTMgr {
            pos: (0, 0),
            wit: Arr::new(ndim),
            ndim,
            storage: vec![0.0; ndim * ndim],
        }
    }

    /// Performs LDL^T factorization from a full matrix.
    ///
    /// $$ A = LDL^T $$
    ///
    /// Returns `true` if the matrix is positive definite.
    pub fn factorize(&mut self, mat: &Arr) -> bool {
        self.factor(|i, j| mat.at(i, j))
    }

    /// Performs LDL^T factorization using lazy element access.
    ///
    /// $$ A = LDL^T $$
    ///
    /// `get_elem(i, j)` returns the matrix element at row `i`, column `j`.
    /// Returns `true` if the matrix is positive definite (all diagonal entries positive).
    pub fn factor(&mut self, get_elem: impl Fn(usize, usize) -> f64) -> bool {
        let start = 0;
        self.pos = (0, 0);
        for i in 0..self.ndim {
            let mut diag = get_elem(i, start);
            for j in start..i {
                let idx_ji = j * self.ndim + i;
                let idx_ij = i * self.ndim + j;
                self.storage[idx_ji] = diag; // keep for later
                let val = diag / self.storage[j * self.ndim + j];
                self.storage[idx_ij] = val; // L[i, j]
                let stop = j + 1;
                // compute diag -= storage[i, start..stop] · storage[start..stop, stop]
                let mut s = 0.0;
                for k in start..stop {
                    s += self.storage[i * self.ndim + k] * self.storage[k * self.ndim + stop];
                }
                diag = get_elem(i, stop) - s;
            }
            self.storage[i * self.ndim + i] = diag;
            if diag <= 0.0 {
                self.pos = (start, i + 1);
                break;
            }
        }
        self.is_spd()
    }

    /// Performs LDL^T factorization allowing for positive semi-definite matrices.
    ///
    /// $$ A = LDL^T, \quad D_{ii} \ge 0 $$
    ///
    /// Returns `true` if the matrix is positive semi-definite (no negative diagonal entries).
    pub fn factor_with_allow_semidefinite(
        &mut self,
        get_elem: impl Fn(usize, usize) -> f64,
    ) -> bool {
        let mut start = 0;
        self.pos = (0, 0);
        for i in 0..self.ndim {
            let mut diag = get_elem(i, start);
            for j in start..i {
                let idx_ji = j * self.ndim + i;
                let idx_ij = i * self.ndim + j;
                self.storage[idx_ji] = diag;
                let val = diag / self.storage[j * self.ndim + j];
                self.storage[idx_ij] = val;
                let stop = j + 1;
                let mut s = 0.0;
                for k in start..stop {
                    s += self.storage[i * self.ndim + k] * self.storage[k * self.ndim + stop];
                }
                diag = get_elem(i, stop) - s;
            }
            self.storage[i * self.ndim + i] = diag;
            if diag < 0.0 {
                self.pos = (start, i + 1);
                break;
            } else if diag == 0.0 {
                start = i + 1;
            }
        }
        self.is_spd()
    }

    /// Checks if the matrix is symmetric positive definite.
    pub fn is_spd(&self) -> bool {
        self.pos.1 == 0
    }

    /// Computes a witness vector proving the matrix is not positive definite.
    ///
    /// $$ v^T A v = -e_p < 0 $$
    ///
    /// Returns the negative eigenvalue `ep`.
    pub fn witness(&mut self) -> f64 {
        assert!(!self.is_spd(), "witness called on SPD matrix");
        let (start, pos) = self.pos;
        let m = pos - 1;
        self.wit[m] = 1.0;
        for i in (start + 1..=m).rev() {
            let mut s = 0.0;
            for k in i..pos {
                s += self.storage[k * self.ndim + (i - 1)] * self.wit[k];
            }
            self.wit[i - 1] = -s;
        }
        -self.storage[m * self.ndim + m]
    }

    /// Computes the quadratic form $$ v^T M v $$ using the witness vector.
    /// `witness()` must be called first.
    pub fn sym_quad(&self, mat: &Arr) -> f64 {
        let (start, end) = self.pos;
        let mut result = 0.0;
        for i in start..end {
            for j in start..end {
                result += self.wit[i] * mat.at(i, j) * self.wit[j];
            }
        }
        result
    }

    /// Computes the upper triangular square root matrix R where $$ A = R^T R $$.
    /// Panics if the matrix is not positive definite.
    pub fn sqrt(&self) -> Arr {
        assert!(self.is_spd(), "sqrt called on non-SPD matrix");
        let mut r = Arr::zeros(self.ndim, self.ndim);
        for i in 0..self.ndim {
            let val = self.storage[i * self.ndim + i].sqrt();
            r.set(i, i, val);
            for j in (i + 1)..self.ndim {
                r.set(i, j, self.storage[j * self.ndim + i] * val);
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    fn arr_from_2d(data: &[&[f64]]) -> Arr {
        let rows = data.len();
        let cols = data[0].len();
        let mut flat = Vec::with_capacity(rows * cols);
        for row in data {
            flat.extend_from_slice(row);
        }
        Arr::with_data(flat, rows, cols)
    }

    fn chol1_matrix() -> Arr {
        arr_from_2d(&[&[25.0, 15.0, -5.0], &[15.0, 18.0, 0.0], &[-5.0, 0.0, 11.0]])
    }

    fn chol2_matrix() -> Arr {
        arr_from_2d(&[
            &[18.0, 22.0, 54.0, 42.0],
            &[22.0, -70.0, 86.0, 62.0],
            &[54.0, 86.0, -174.0, 134.0],
            &[42.0, 62.0, 134.0, -106.0],
        ])
    }

    fn chol3_matrix() -> Arr {
        arr_from_2d(&[&[0.0, 15.0, -5.0], &[15.0, 18.0, 0.0], &[-5.0, 0.0, 11.0]])
    }

    fn chol7_matrix() -> Arr {
        arr_from_2d(&[&[0.0, 15.0, -5.0], &[15.0, 18.0, 0.0], &[-5.0, 0.0, -20.0]])
    }

    fn chol8_matrix() -> Arr {
        arr_from_2d(&[&[0.0, 15.0, -5.0], &[15.0, 18.0, 0.0], &[-5.0, 0.0, 20.0]])
    }

    #[test]
    fn test_chol1() {
        let m = chol1_matrix();
        let mut ldlt = LDLTMgr::new(3);
        assert!(ldlt.factorize(&m));
    }

    #[test]
    fn test_chol2() {
        let m = chol2_matrix();
        let mut ldlt = LDLTMgr::new(4);
        assert!(!ldlt.factorize(&m));
        ldlt.witness();
        assert_eq!(ldlt.pos, (0, 2));
    }

    #[test]
    fn test_chol3() {
        let m = chol3_matrix();
        let mut ldlt = LDLTMgr::new(3);
        assert!(!ldlt.factorize(&m));
        let ep = ldlt.witness();
        assert_eq!(ldlt.pos, (0, 1));
        assert_approx_eq!(ldlt.wit[0], 1.0);
        assert_approx_eq!(ep, 0.0);
    }

    #[test]
    fn test_chol4() {
        let m = chol1_matrix();
        let mut ldlt = LDLTMgr::new(3);
        assert!(ldlt.factorize(&m));
    }

    #[test]
    fn test_chol5() {
        let m = chol2_matrix();
        let mut ldlt = LDLTMgr::new(4);
        assert!(!ldlt.factorize(&m));
        ldlt.witness();
        assert_eq!(ldlt.pos, (0, 2));
    }

    #[test]
    fn test_chol6() {
        let m = chol3_matrix();
        let mut ldlt = LDLTMgr::new(3);
        assert!(ldlt.factor_with_allow_semidefinite(|i, j| m.at(i, j)));
    }

    #[test]
    fn test_chol7() {
        let m = chol7_matrix();
        let mut ldlt = LDLTMgr::new(3);
        assert!(!ldlt.factor_with_allow_semidefinite(|i, j| m.at(i, j)));
        let ep = ldlt.witness();
        assert_approx_eq!(ep, 20.0);
    }

    #[test]
    fn test_chol8() {
        let m = chol8_matrix();
        let mut ldlt = LDLTMgr::new(3);
        assert!(!ldlt.factorize(&m));
    }

    #[test]
    fn test_chol9() {
        let m = chol8_matrix();
        let mut ldlt = LDLTMgr::new(3);
        assert!(ldlt.factor_with_allow_semidefinite(|i, j| m.at(i, j)));
    }

    #[test]
    fn test_ldlt_mgr_sqrt() {
        let m = arr_from_2d(&[&[1.0, 0.5, 0.5], &[0.5, 1.25, 0.75], &[0.5, 0.75, 1.5]]);
        let mut ldlt = LDLTMgr::new(3);
        ldlt.factor(|i, j| m.at(i, j));
        assert!(ldlt.is_spd());
        let r = ldlt.sqrt();
        let expected = arr_from_2d(&[&[1.0, 0.5, 0.5], &[0.0, 1.0, 0.5], &[0.0, 0.0, 1.0]]);
        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(r.at(i, j), expected.at(i, j));
            }
        }
    }
}
