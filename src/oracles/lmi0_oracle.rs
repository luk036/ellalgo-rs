use crate::arr::Arr;
use crate::oracles::ldlt_mgr::LDLTMgr;
use crate::oracles::lmi_oracle_base::assess_feas_impl;

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
        assess_feas_impl(&mut self.ldlt_mgr, &self.mat_f, x, -1.0, |i, j| {
            let mut s = 0.0;
            for k in 0..n {
                s += self.mat_f[k].at(i, j) * x[k];
            }
            s
        })
    }
}
