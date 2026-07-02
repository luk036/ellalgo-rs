use crate::arr::Arr;
use crate::cutting_plane::{OracleFeas, SingleCut};
use crate::oracles::ldlt_mgr::LDLTMgr;

pub struct LMIOracle {
    mat_f: Vec<Arr>,
    mat_b: Arr,
    ldlt_mgr: LDLTMgr,
}

impl LMIOracle {
    pub fn new(mat_f: Vec<Arr>, mat_b: Arr) -> Self {
        let ndim = mat_b.rows();
        let ldlt_mgr = LDLTMgr::new(ndim);
        LMIOracle {
            mat_f,
            mat_b,
            ldlt_mgr,
        }
    }
}

impl OracleFeas<Arr> for LMIOracle {
    type CutChoice = SingleCut;

    /// Assess LMI feasibility: $$ F(x) = B - \sum_{i=1}^{n} x_i F_i \succ 0 $$
    ///
    /// Returns `None` if $$ F(x) \succ 0 $$ (feasible).
    /// Otherwise returns the gradient and offset.
    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, SingleCut)> {
        let n = xc.len();
        let feas = self.ldlt_mgr.factor(|i, j| {
            let mut s = self.mat_b.at(i, j);
            for k in 0..n {
                s -= self.mat_f[k].at(i, j) * xc[k];
            }
            s
        });
        if feas {
            return None;
        }
        let ep = self.ldlt_mgr.witness();
        let mut g = Arr::new(n);
        for k in 0..n {
            g[k] = self.ldlt_mgr.sym_quad(&self.mat_f[k]);
        }
        Some((g, SingleCut(ep)))
    }
}
