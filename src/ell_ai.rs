use ndarray::Array2;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use std::cmp::Ordering;
use std::ops::Mul;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Div;
use std::ops::Neg;
use std::f64::EPSILON;
use std::f64::INFINITY;
use std::f64::consts::PI;

type ArrayType = Array1<f64>;
type CutChoice = f64;
type Cut = (ArrayType, CutChoice);

pub trait SearchSpace {
    fn _update_core(&mut self, cut: Cut, cut_strategy: fn(CutChoice, f64) -> (CutChoice, f64, f64, f64)) -> CutStatus;
}

pub trait SearchSpaceQ {
    fn _update_core_q(&mut self, cut: Cut, cut_strategy: fn(CutChoice, f64) -> (CutChoice, f64, f64, f64)) -> CutStatus;
}

#[derive(PartialEq)]
pub enum CutStatus {
    Success,
    Infeasible,
    Unbounded,
}

pub struct Ell {
    no_defer_trick: bool,
    _mq: Array2<f64>,
    _xc: ArrayType,
    _kappa: f64,
    _tsq: f64,
    _helper: EllCalc,
}

impl SearchSpace for Ell {}
impl SearchSpaceQ for Ell {}

impl Ell {
    pub fn new(val: f64, xc: ArrayType) -> Self {
        let ndim = xc.len();
        let helper = EllCalc::new(ndim);
        let mut mq = Array2::eye(ndim);
        let mut kappa = 1.0;
        let tsq = 0.0;
        if val.is_finite() {
            kappa = val;
            mq = Array2::eye(ndim);
        } else {
            kappa = 1.0;
            mq = Array2::from_diag(&xc);
        }
        Ell {
            no_defer_trick: false,
            _mq: mq,
            _xc: xc,
            _kappa: kappa,
            _tsq: tsq,
            _helper: helper,
        }
    }

    fn _update_core(&mut self, cut: Cut, cut_strategy: fn(CutChoice, f64) -> (CutChoice, f64, f64, f64)) -> CutStatus {
        let (grad, beta) = cut;
        let grad_t = self._mq.dot(&grad);
        let omega = grad.dot(&grad_t);
        self._tsq = self._kappa * omega;
        let (status, rho, sigma, delta) = cut_strategy(beta, self._tsq);
        if status != CutStatus::Success {
            return status;
        }
        self._xc -= &grad_t * (rho / omega);
        self._mq -= &grad_t.outer(&grad_t) * (sigma / omega);
        self._kappa *= delta;
        if self.no_defer_trick {
            self._mq *= self._kappa;
            self._kappa = 1.0;
        }
        status
    }
}

