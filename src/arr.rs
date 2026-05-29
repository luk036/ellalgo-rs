//! Minimal flat-vector array type replacing `ndarray` for small optimization problems.
//!
//! Supports 1D (vector) and 2D (row-major matrix) with operations
//! needed by ellipsoid-method-based solvers.

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A flat-vector array that can be 1D (vector) or 2D (row-major matrix).
#[derive(Debug, Clone)]
pub struct Arr {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Arr {
    pub fn new(n: usize) -> Self {
        Arr {
            data: vec![0.0; n],
            rows: n,
            cols: 0,
        }
    }
    pub fn zeros(r: usize, c: usize) -> Self {
        Arr {
            data: vec![0.0; r * c],
            rows: r,
            cols: c,
        }
    }
    pub fn full(r: usize, c: usize, val: f64) -> Self {
        Arr {
            data: vec![val; r * c],
            rows: r,
            cols: c,
        }
    }
    pub fn eye(n: usize) -> Self {
        let mut a = Arr::zeros(n, n);
        for i in 0..n {
            a.set(i, i, 1.0);
        }
        a
    }
    pub fn from_diag(v: &Arr) -> Self {
        assert!(!v.is_2d());
        let n = v.len();
        let mut a = Arr::zeros(n, n);
        for i in 0..n {
            a.set(i, i, v[i]);
        }
        a
    }
    pub fn from(v: Vec<f64>) -> Self {
        let n = v.len();
        Arr {
            data: v,
            rows: n,
            cols: 0,
        }
    }
    pub fn ones(n: usize) -> Self {
        Arr {
            data: vec![1.0; n],
            rows: n,
            cols: 0,
        }
    }
    pub fn from_fn(n: usize, f: impl FnMut(usize) -> f64) -> Self {
        let data: Vec<f64> = (0..n).map(f).collect();
        Arr {
            data,
            rows: n,
            cols: 0,
        }
    }

    /// Construct a 2D matrix from a flat vector in row-major order.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    pub fn from_shape_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Arr { data, rows, cols }
    }

    /// Extract a row as a 1D array.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.rows` or if the array is not 2D.
    pub fn row(&self, i: usize) -> Self {
        assert!(self.is_2d());
        assert!(i < self.rows);
        let start = i * self.cols;
        Arr {
            data: self.data[start..start + self.cols].to_vec(),
            rows: self.cols,
            cols: 0,
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.data.len()
    }
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    pub fn is_2d(&self) -> bool {
        self.cols > 0
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.data.iter()
    }
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f64> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(self.is_2d());
        self.data[i * self.cols + j]
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        debug_assert!(self.is_2d());
        self.data[i * self.cols + j] = val;
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.get(i, j)
    }
    #[inline]
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

impl Index<usize> for Arr {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}
impl IndexMut<usize> for Arr {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}
impl Index<(usize, usize)> for Arr {
    type Output = f64;
    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.data[r * self.cols + c]
    }
}
impl IndexMut<(usize, usize)> for Arr {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }
}

impl PartialEq for Arr {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl From<Vec<f64>> for Arr {
    fn from(v: Vec<f64>) -> Self {
        Arr::from(v)
    }
}
impl From<&[f64]> for Arr {
    fn from(s: &[f64]) -> Self {
        Arr::from(s.to_vec())
    }
}

impl AddAssign<f64> for Arr {
    #[inline]
    fn add_assign(&mut self, s: f64) {
        for v in &mut self.data {
            *v += s;
        }
    }
}
impl SubAssign<f64> for Arr {
    #[inline]
    fn sub_assign(&mut self, s: f64) {
        for v in &mut self.data {
            *v -= s;
        }
    }
}
impl MulAssign<f64> for Arr {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        for v in &mut self.data {
            *v *= s;
        }
    }
}
impl DivAssign<f64> for Arr {
    #[inline]
    fn div_assign(&mut self, s: f64) {
        let inv = 1.0 / s;
        for v in &mut self.data {
            *v *= inv;
        }
    }
}

impl AddAssign<&Arr> for Arr {
    #[inline]
    fn add_assign(&mut self, o: &Arr) {
        assert_eq!(self.size(), o.size());
        for (a, b) in self.data.iter_mut().zip(o.data.iter()) {
            *a += *b;
        }
    }
}
impl SubAssign<&Arr> for Arr {
    #[inline]
    fn sub_assign(&mut self, o: &Arr) {
        assert_eq!(self.size(), o.size());
        for (a, b) in self.data.iter_mut().zip(o.data.iter()) {
            *a -= *b;
        }
    }
}

impl Add<&Arr> for Arr {
    type Output = Arr;
    #[inline]
    fn add(mut self, o: &Arr) -> Arr {
        self += o;
        self
    }
}
impl Sub<&Arr> for Arr {
    type Output = Arr;
    #[inline]
    fn sub(mut self, o: &Arr) -> Arr {
        self -= o;
        self
    }
}
impl Mul<&Arr> for Arr {
    type Output = Arr;
    #[inline]
    fn mul(mut self, o: &Arr) -> Arr {
        assert_eq!(self.size(), o.size());
        for (a, b) in self.data.iter_mut().zip(o.data.iter()) {
            *a *= *b;
        }
        self
    }
}

impl Add<&Arr> for &Arr {
    type Output = Arr;
    #[inline]
    fn add(self, o: &Arr) -> Arr {
        assert_eq!(self.size(), o.size());
        Arr {
            data: self
                .data
                .iter()
                .zip(o.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}
impl Sub<&Arr> for &Arr {
    type Output = Arr;
    #[inline]
    fn sub(self, o: &Arr) -> Arr {
        assert_eq!(self.size(), o.size());
        Arr {
            data: self
                .data
                .iter()
                .zip(o.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}
impl Mul<&Arr> for &Arr {
    type Output = Arr;
    #[inline]
    fn mul(self, o: &Arr) -> Arr {
        assert_eq!(self.size(), o.size());
        Arr {
            data: self
                .data
                .iter()
                .zip(o.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Mul<f64> for &Arr {
    type Output = Arr;
    #[inline]
    fn mul(self, s: f64) -> Arr {
        let mut a = self.clone();
        a *= s;
        a
    }
}
impl Mul<f64> for Arr {
    type Output = Arr;
    #[inline]
    fn mul(mut self, s: f64) -> Arr {
        self *= s;
        self
    }
}
impl Mul<&Arr> for f64 {
    type Output = Arr;
    #[inline]
    fn mul(self, a: &Arr) -> Arr {
        let mut r = a.clone();
        r *= self;
        r
    }
}
impl Mul<Arr> for f64 {
    type Output = Arr;
    #[inline]
    fn mul(self, mut a: Arr) -> Arr {
        a *= self;
        a
    }
}

impl Div<f64> for &Arr {
    type Output = Arr;
    #[inline]
    fn div(self, s: f64) -> Arr {
        let mut a = self.clone();
        a /= s;
        a
    }
}
impl Div<f64> for Arr {
    type Output = Arr;
    #[inline]
    fn div(mut self, s: f64) -> Arr {
        self /= s;
        self
    }
}

impl Neg for Arr {
    type Output = Arr;
    #[inline]
    fn neg(mut self) -> Arr {
        for v in &mut self.data {
            *v = -(*v);
        }
        self
    }
}
impl Neg for &Arr {
    type Output = Arr;
    #[inline]
    fn neg(self) -> Arr {
        let mut a = self.clone();
        for v in &mut a.data {
            *v = -(*v);
        }
        a
    }
}

impl Arr {
    pub fn dot_mv(&self, x: &Arr) -> Arr {
        assert!(self.is_2d());
        assert!(!x.is_2d());
        assert_eq!(self.cols, x.len());
        let mut out = vec![0.0; self.rows];
        for (i, out_i) in out.iter_mut().enumerate().take(self.rows) {
            let rs = i * self.cols;
            for j in 0..self.cols {
                *out_i += self.data[rs + j] * x[j];
            }
        }
        Arr {
            data: out,
            rows: self.rows,
            cols: 0,
        }
    }
    pub fn dot(&self, other: &Arr) -> f64 {
        assert!(!self.is_2d() && !other.is_2d());
        assert_eq!(self.len(), other.len());
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    pub fn outer(&self, other: &Arr) -> Arr {
        assert!(!self.is_2d() && !other.is_2d());
        let m = self.len();
        let n = other.len();
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            let rs = i * n;
            for j in 0..n {
                data[rs + j] = self[i] * other[j];
            }
        }
        Arr {
            data,
            rows: m,
            cols: n,
        }
    }
    pub fn rank_one_update(&mut self, alpha: f64, u: &Arr) {
        assert!(self.is_2d());
        assert!(!u.is_2d());
        let n = self.rows;
        assert_eq!(n, u.len());
        for i in 0..n {
            let ui = u[i];
            let rs = i * n;
            for j in 0..=i {
                let uj = u[j];
                self.data[rs + j] += alpha * ui * uj;
                if i != j {
                    self.data[j * n + i] = self.data[rs + j];
                }
            }
        }
    }
    pub fn copy_from(&mut self, other: &Arr) {
        assert_eq!(self.size(), other.size());
        self.data.copy_from_slice(&other.data);
    }
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }
    pub fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(f64) -> f64,
    {
        let data: Vec<f64> = self.data.iter().copied().map(f).collect();
        Arr {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

pub fn linspace(start: f64, end: f64, n: usize) -> Arr {
    let mut a = Arr::new(n);
    if n == 0 {
        return a;
    }
    if n == 1 {
        a[0] = start;
        return a;
    }
    let step = (end - start) / (n - 1) as f64;
    for i in 0..n {
        a[i] = start + step * i as f64;
    }
    a
}

pub fn arange(start: f64, end: f64) -> Arr {
    let n = if end > start {
        (end - start) as usize
    } else {
        0
    };
    let mut a = Arr::new(n);
    for i in 0..n {
        a[i] = start + i as f64;
    }
    a
}

pub fn make_same_shape(a: &Arr) -> Arr {
    if a.is_2d() {
        Arr::zeros(a.rows(), a.cols())
    } else {
        Arr::new(a.rows())
    }
}

pub fn cos(a: &Arr) -> Arr {
    let mut o = make_same_shape(a);
    for i in 0..a.size() {
        o[i] = a[i].cos();
    }
    o
}
pub fn ln(a: &Arr) -> Arr {
    let mut o = make_same_shape(a);
    for i in 0..a.size() {
        o[i] = a[i].ln();
    }
    o
}
pub fn abs(a: &Arr) -> Arr {
    let mut o = make_same_shape(a);
    for i in 0..a.size() {
        o[i] = a[i].abs();
    }
    o
}
pub fn exp(a: &Arr) -> Arr {
    let mut o = make_same_shape(a);
    for i in 0..a.size() {
        o[i] = a[i].exp();
    }
    o
}
pub fn sqrt(a: &Arr) -> Arr {
    let mut o = make_same_shape(a);
    for i in 0..a.size() {
        o[i] = a[i].sqrt();
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_1d() {
        let a = Arr::new(5);
        assert_eq!(a.size(), 5);
        assert!(!a.is_2d());
        assert_eq!(a[0], 0.0);
    }
    #[test]
    fn test_zeros_2d() {
        let a = Arr::zeros(3, 4);
        assert!(a.is_2d());
        assert_eq!(a.get(0, 0), 0.0);
        assert_eq!(a.get(2, 3), 0.0);
    }
    #[test]
    fn test_full() {
        let a = Arr::full(2, 3, 1.5);
        assert_eq!(a.get(0, 0), 1.5);
        assert_eq!(a.get(1, 2), 1.5);
    }
    #[test]
    fn test_eye() {
        let i = Arr::eye(4);
        assert!(i.is_2d());
        for j in 0..4 {
            assert_eq!(i.get(j, j), 1.0);
        }
        assert_eq!(i.get(0, 1), 0.0);
    }
    #[test]
    fn test_from_vec() {
        let a = Arr::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(a[0], 1.0);
        assert_eq!(a[2], 3.0);
    }
    #[test]
    fn test_from_fn() {
        let a = Arr::from_fn(4, |i| i as f64 * 2.0);
        assert_eq!(a[3], 6.0);
    }
    #[test]
    fn test_ones() {
        let a = Arr::ones(4);
        assert_eq!(a[0], 1.0);
        assert_eq!(a[3], 1.0);
    }
    #[test]
    fn test_index_mut() {
        let mut a = Arr::new(3);
        a[0] = 10.0;
        assert_eq!(a[0], 10.0);
    }
    #[test]
    fn test_index_2d_tuple() {
        let mut a = Arr::zeros(2, 3);
        a[(0, 0)] = 1.0;
        assert_eq!(a.get(0, 0), 1.0);
    }
    #[test]
    fn test_scalar_ops() {
        let mut a = Arr::from(vec![1.0, 2.0, 3.0]);
        a += 1.0;
        assert_eq!(a, Arr::from(vec![2.0, 3.0, 4.0]));
        a *= 2.0;
        assert_eq!(a, Arr::from(vec![4.0, 6.0, 8.0]));
    }
    #[test]
    fn test_add_arr() {
        assert_eq!(
            &Arr::from(vec![1.0, 2.0]) + &Arr::from(vec![3.0, 4.0]),
            Arr::from(vec![4.0, 6.0])
        );
    }
    #[test]
    fn test_sub_arr() {
        assert_eq!(
            &Arr::from(vec![5.0, 7.0]) - &Arr::from(vec![1.0, 2.0]),
            Arr::from(vec![4.0, 5.0])
        );
    }
    #[test]
    fn test_mul_arr() {
        assert_eq!(
            &Arr::from(vec![2.0, 3.0]) * &Arr::from(vec![5.0, 6.0]),
            Arr::from(vec![10.0, 18.0])
        );
    }
    #[test]
    fn test_scalar_mul() {
        assert_eq!(&Arr::from(vec![1.0, 2.0]) * 2.0, Arr::from(vec![2.0, 4.0]));
        assert_eq!(3.0 * &Arr::from(vec![1.0, 2.0]), Arr::from(vec![3.0, 6.0]));
    }
    #[test]
    fn test_dot_mv() {
        let m = Arr::eye(3);
        let y = m.dot_mv(&Arr::from(vec![2.0, 3.0, 4.0]));
        assert_eq!(y[0], 2.0);
    }
    #[test]
    fn test_dot() {
        assert_eq!(
            Arr::from(vec![1.0, 2.0, 3.0]).dot(&Arr::from(vec![4.0, 5.0, 6.0])),
            32.0
        );
    }
    #[test]
    fn test_outer() {
        let o = Arr::from(vec![1.0, 2.0]).outer(&Arr::from(vec![3.0, 4.0, 5.0]));
        assert_eq!(o.get(0, 0), 3.0);
        assert_eq!(o.get(1, 2), 10.0);
    }
    #[test]
    fn test_rank_one_update() {
        let mut m = Arr::eye(3);
        m.rank_one_update(-1.0, &Arr::from(vec![1.0, 2.0, 3.0]));
        assert_eq!(m.get(0, 0), 0.0);
        assert_eq!(m.get(1, 0), -2.0);
    }
    #[test]
    fn test_sum() {
        assert_eq!(Arr::from(vec![1.0, 2.0, 3.0, 4.0]).sum(), 10.0);
    }
    #[test]
    fn test_linspace() {
        let a = linspace(0.0, 10.0, 5);
        assert_eq!(a[2], 5.0);
    }
    #[test]
    fn test_arange() {
        let a = arange(0.0, 5.0);
        assert_eq!(a[4], 4.0);
    }
    #[test]
    fn test_eye_large() {
        let i = Arr::eye(50);
        assert_eq!(i.get(0, 0), 1.0);
        assert_eq!(i.get(0, 49), 0.0);
    }
}
