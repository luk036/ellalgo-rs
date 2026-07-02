use crate::arr::Arr;
use crate::oracles::ldlt_mgr::LDLTMgr;

pub struct LMI0Oracle {
    mat_f: Vec<Arr>,
    ldlt_mgr: LDLTMgr,
}

impl LMI0Oracle {
    pub fn new(mat_f: Vec<Arr>) -> Self {
        let ndim = mat_f[0].rows();
        let ldlt_mgr = LDLTMgr::new(ndim);
        LMI0Oracle { mat_f, ldlt_mgr }
    }

    /// Assess LMI feasibility: $$ F(x) = \sum_{i=1}^{n} x_i F_i \succ 0 $$
    ///
    /// Returns `None` if $$ F(x) \succ 0 $$ (feasible).
    /// Otherwise returns the gradient and offset.
    pub fn assess_feas(&mut self, x: &Arr) -> Option<(Arr, f64)> {
        let n = x.len();
        let feas = self.ldlt_mgr.factor(|i, j| {
            let mut s = 0.0;
            for k in 0..n {
                s += self.mat_f[k].at(i, j) * x[k];
            }
            s
        });
        if feas {
            return None;
        }
        let ep = self.ldlt_mgr.witness();
        let mut g = Arr::new(n);
        for k in 0..n {
            g[k] = -self.ldlt_mgr.sym_quad(&self.mat_f[k]);
        }
        Some((g, ep))
    }
}
