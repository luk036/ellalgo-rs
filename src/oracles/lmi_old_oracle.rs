use crate::arr::Arr;
use crate::cutting_plane::{OracleFeas, SingleCut};
use crate::oracles::ldlt_mgr::LDLTMgr;

pub struct LMIOldOracle {
    mat_f: Vec<Arr>,
    mat_b: Arr,
    ldlt_mgr: LDLTMgr,
}

impl LMIOldOracle {
    pub fn new(mat_f: Vec<Arr>, mat_b: Arr) -> Self {
        let ndim = mat_b.rows();
        let ldlt_mgr = LDLTMgr::new(ndim);
        LMIOldOracle {
            mat_f,
            mat_b,
            ldlt_mgr,
        }
    }
}

impl OracleFeas<Arr> for LMIOldOracle {
    type CutChoice = SingleCut;

    fn assess_feas(&mut self, xc: &Arr) -> Option<(Arr, SingleCut)> {
        let n = xc.len();
        let ndim = self.mat_b.rows();
        let mut a = self.mat_b.clone();
        for k in 0..n {
            let xk = xc[k];
            for i in 0..ndim {
                for j in 0..ndim {
                    let old = a.at(i, j);
                    let new = old - self.mat_f[k].at(i, j) * xk;
                    a.set(i, j, new);
                }
            }
        }
        if self.ldlt_mgr.factorize(&a) {
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
