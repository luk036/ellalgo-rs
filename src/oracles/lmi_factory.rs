//! Factory functions for the LMI oracle family.
//!
//! Provides uniform construction entry points for the three LMI oracle
//! variants: [`LMIOracle`] (lazy), [`LMI0Oracle`] (no constant term), and
//! [`LMIOldOracle`] (explicit).

use crate::arr::Arr;
use crate::oracles::lmi0_oracle::LMI0Oracle;
use crate::oracles::lmi_old_oracle::LMIOldOracle;
use crate::oracles::lmi_oracle::LMIOracle;

/// Create an [`LMIOracle`] (lazy matrix form).
///
/// # Arguments
///
/// * `mat_f` - List of symmetric coefficient matrices `[F₁, F₂, ..., Fₙ]`
/// * `mat_b` - Constant matrix `B` defining the LMI constraint `B − ΣFₖxₖ ⪰ 0`
#[inline]
pub fn make_lmi_oracle(mat_f: Vec<Arr>, mat_b: Arr) -> LMIOracle {
    LMIOracle::new(mat_f, mat_b)
}

/// Create an [`LMI0Oracle`] (compact form, no constant term).
///
/// # Arguments
///
/// * `mat_f` - List of symmetric coefficient matrices `[F₁, F₂, ..., Fₙ]`
#[inline]
pub fn make_lmi0_oracle(mat_f: Vec<Arr>) -> LMI0Oracle {
    LMI0Oracle::new(mat_f)
}

/// Create an [`LMIOldOracle`] (explicit matrix form).
///
/// # Arguments
///
/// * `mat_f` - List of symmetric coefficient matrices `[F₁, F₂, ..., Fₙ]`
/// * `mat_b` - Constant matrix `B` defining the LMI constraint `B − ΣFₖxₖ ⪰ 0`
#[inline]
pub fn make_lmi_old_oracle(mat_f: Vec<Arr>, mat_b: Arr) -> LMIOldOracle {
    LMIOldOracle::new(mat_f, mat_b)
}
