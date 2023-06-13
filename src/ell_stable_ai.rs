use ndarray::Array2;
use ndarray::Array1;
use ndarray::Array;
use ndarray::Axis;
use std::cmp::Ordering;
use std::ops::Mul;
use std::ops::Sub;
use std::ops::Div;
use std::ops::Add;
use std::ops::Neg;
use std::clone::Clone;
use std::marker::Copy;
use std::convert::From;
use std::convert::Into;
use std::option::Option;
use std::result::Result;

type Matrix = Array2<f64>;
type ArrayType = Array1<f64>;
type CutChoice = f64;
type Cut = (ArrayType, CutChoice);

enum CutStatus {
    Success,
    Failure,
    Incomplete,
}

struct EllCalc {
    // implementation details
}

struct EllStable {
    no_defer_trick: bool,
    _mq: Matrix,
    _xc: ArrayType,
    _kappa: f64,
    _tsq: f64,
    _n: usize,
    _helper: EllCalc,
}

impl EllStable {
    fn _update_core<F>(&mut self, cut: Cut, cut_strategy: F) -> CutStatus
    where
        F: Fn(CutChoice, f64) -> (CutStatus, f64, f64, f64),
    {
        let (g, beta) = cut;
        let mut invLg = g.to_owned();
        for j in 0..self._n - 1 {
            for i in j + 1..self._n {
                self._mq[[j, i]] = self._mq[[i, j]] * invLg[j];
                invLg[i] -= self._mq[[j, i]];
            }
        }
        let mut invDinvLg = invLg.to_owned();
        for i in 0..self._n {
            invDinvLg[i] *= self._mq[[i, i]];
        }
        let gg_t = invLg.mul(&invDinvLg);
        let omega = gg_t.sum();
        self._tsq = self._kappa * omega;
        let (status, rho, sigma, delta) = cut_strategy(beta, self._tsq);
        let mut g_t = invDinvLg.to_owned();
        for i in (1..self._n).rev() {
            for j in i..self._n {
                g_t[i - 1] -= self._mq[[j, i - 1]] * g_t[j];
            }
        }
        let mu = sigma / (1.0 - sigma);
        let oldt = omega / mu;
        let mut v = g.to_owned();
        self._kappa *= delta;
        if self.no_defer_trick {
            self._mq *= self._kappa;
            self._kappa = 1.0;
        }
        status
    }
}

