//! Round-robin index helper for cyclic constraint scanning.
//!
//! Extracts the repeated `idx += 1; if idx == N { idx = 0 }` idiom into a
//! small stateful helper (used by `LowpassOracle` and `ProfitOracle`).

/// Round-robin index generator over a half-open range `[lo, hi)`.
///
/// Successive calls to [`advance`](RoundRobin::advance) yield
/// `lo, lo+1, ..., hi-1, lo, ...`. The cursor starts at `lo - 1` so the first
/// call returns `lo`, matching the `idx += 1; if idx == N { idx = 0 }` idiom
/// it replaces.
///
/// # Examples
///
/// ```
/// use ellalgo_rs::round_robin::RoundRobin;
///
/// let mut rr = RoundRobin::new(3);
/// assert_eq!(rr.advance(), 0);
/// assert_eq!(rr.advance(), 1);
/// assert_eq!(rr.advance(), 2);
/// assert_eq!(rr.advance(), 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundRobin {
    cur: i32,
    lo: i32,
    hi: i32,
}

impl RoundRobin {
    /// Round-robin over `[0, hi)`.
    #[inline]
    pub fn new(hi: i32) -> Self {
        Self::new_range(0, hi)
    }

    /// Round-robin over `[lo, hi)`.
    pub fn new_range(lo: i32, hi: i32) -> Self {
        RoundRobin {
            cur: lo - 1,
            lo,
            hi,
        }
    }

    /// Advance to the next index and return it.
    pub fn advance(&mut self) -> i32 {
        self.cur += 1;
        if self.cur == self.hi {
            self.cur = self.lo;
        }
        self.cur
    }

    /// The current index (as last returned by `advance`).
    #[inline]
    pub fn current(&self) -> i32 {
        self.cur
    }
}
