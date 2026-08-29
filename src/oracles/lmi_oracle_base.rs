//! Shared skeleton for LMI feasibility oracles (Template Method).
//!
//! The three LMI oracle flavors (LMIOracle, LMI0Oracle, LMIOldOracle) differ
//! only in how the matrix elements `A(i,j)` are assembled (lazy element
//! access vs. an eagerly-built matrix) and in the `sym_quad` sign convention.
//! The factorization / witness / cut-packing pipeline is identical and lives
//! here.

use crate::arr::Arr;
use crate::oracles::ldlt_mgr::LDLTMgr;

/// Shared assess_feas skeleton: factor, witness, sym_quad, pack cut.
///
/// Returns `None` if `A(x)` is positive definite (feasible); otherwise a cut
/// tuple `(g, ep)` where `g` is the subgradient and `ep` the violation.
///
/// # Arguments
///
/// * `ldlt_mgr` - LDL^T factorization manager (mutable for state)
/// * `mat_f` - Coefficient matrices `F_k`
/// * `xc` - Evaluation point
/// * `sign` - `+1` or `-1` convention for the sym_quad subgradient
/// * `get_elem` - Lazy accessor for matrix element `A(i, j)`
pub(crate) fn assess_feas_impl(
    ldlt_mgr: &mut LDLTMgr,
    mat_f: &[Arr],
    xc: &Arr,
    sign: f64,
    get_elem: impl Fn(usize, usize) -> f64,
) -> Option<(Arr, f64)> {
    let n = xc.len();
    if ldlt_mgr.factor(get_elem) {
        return None; // Matrix is PSD => feasible solution
    }
    // If infeasible, compute cut information:
    let ep = ldlt_mgr.witness(); // Witness vector for negative eigenvalue
    let mut g = Arr::new(n);
    for k in 0..n {
        g[k] = sign * ldlt_mgr.sym_quad(&mat_f[k]);
    }
    Some((g, ep))
}
