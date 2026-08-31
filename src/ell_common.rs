//! Shared `Ell` / `EllStable` boilerplate (Strategy pattern).
//!
//! `Ell` (classic direct Q-update) and `EllStable` (LDL^T update) expose the
//! same public API: identical constructors and an identical [`SearchSpace`]
//! impl. Only the private `update_core` differs — that is the strategy the two
//! types select by construction. The shared [`SearchSpace`] impl is generated
//! here by a macro so the two types stay in lockstep.

/// Generate the [`SearchSpace`] impl for an ellipsoid type.
macro_rules! impl_search_space {
    ($t:ty) => {
        impl crate::cutting_plane::SearchSpace for $t {
            type ArrayType = crate::arr::Arr;

            #[inline]
            fn xc(&self) -> &Self::ArrayType {
                &self.xc
            }

            #[inline]
            fn tsq(&self) -> f64 {
                self.tsq
            }

            fn update_bias_cut<T>(
                &mut self,
                cut: &(Self::ArrayType, T),
            ) -> crate::cutting_plane::CutStatus
            where
                T: crate::cutting_plane::UpdateByCutChoice<Self, ArrayType = Self::ArrayType>,
            {
                let (grad, beta) = cut;
                beta.update_bias_cut_by(self, grad)
            }

            fn update_central_cut<T>(
                &mut self,
                cut: &(Self::ArrayType, T),
            ) -> crate::cutting_plane::CutStatus
            where
                T: crate::cutting_plane::UpdateByCutChoice<Self, ArrayType = Self::ArrayType>,
            {
                let (grad, beta) = cut;
                beta.update_central_cut_by(self, grad)
            }

            fn update_q<T>(&mut self, cut: &(Self::ArrayType, T)) -> crate::cutting_plane::CutStatus
            where
                T: crate::cutting_plane::UpdateByCutChoice<Self, ArrayType = Self::ArrayType>,
            {
                let (grad, beta) = cut;
                beta.update_q_by(self, grad)
            }

            #[inline]
            fn set_xc(&mut self, x: Self::ArrayType) {
                self.xc = x;
            }
        }
    };
}

pub(crate) use impl_search_space;
