    fn update<T: UpdateByCutChoices>(&mut self, cut: (Arr, T)) -> (CutStatus, f64) {
        let (grad, beta) = cut;
        // calculate inv(L)*grad: (n-1)*n/2 multiplications
        let mut inv_ml_g = grad.clone(); // initial x0
        for i in 0..self.n {
            for j in 0..i {
                self.mq[[i, j]] = self.mq[[j, i]] * inv_ml_g[j];
                // keep for rank-one update
                inv_ml_g[i] -= self.mq[[i, j]];
            }
        }

        // calculate inv(D)*inv(L)*grad: n
        let mut inv_md_inv_ml_g = inv_ml_g.clone();  // initially
        for i in 0..self.n {
            inv_md_inv_ml_g[i] *= self.mq[[i, i]];
        }

        // calculate omega: n
        let mut g_mq_g = inv_md_inv_ml_g.clone();  // initially
        let mut omega = 0.0;     // initially
        for i in 0..self.n {
            g_mq_g[i] *= inv_ml_g[i];
            omega += g_mq_g[i];
        }

        self.tsq = self.kappa * omega;
        let mut status = self.update_cut(beta);
        if status != CutStatus::Success {
            return (status, self.tsq);
        }

        // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad : (n-1)*n/2
        let mut mq_g = inv_md_inv_ml_g.clone();  // initially
        for i in (1..self.n).rev() { // backward subsituition
            for j in i..self.n {
                mq_g[i - 1] -= self.mq[[i, j]] * mq_g[j];  // ???
            }
        }

        // calculate xc: n
        self.xc -= &((self.helper.rho / omega) * &mq_g); // n

        // rank-one update: 3*n + (n-1)*n/2
        // let r = self.sigma / omega;
        let mu = self.sigma / (1.0 - self.sigma);
        let mut oldt = omega / mu;  // initially
        let m = self.n - 1;
        for j in 0..self.m {
            // p=sqrt(k)*vv[j];
            // let p = inv_ml_g[j];
            // let mup = mu * p;
            let t = oldt + g_mq_g[j];
            // self.mq[[j, j]] /= t; // update invD
            let beta2 = inv_md_inv_ml_g[j] / t;
            self.mq[[j, j]] *= oldt / t;  // update invD
            for l in (j + 1)..self.n {
                // v(l) -= p * self.mq(j, l);
                self.mq[[j, l]] += beta2 * self.mq[[l, j]];
            }
            oldt = t;
        }

        // let p = inv_ml_g(n1);
        // let mup = mu * p;
        let t = oldt + g_mq_g[m];
        self.mq[[m, m]] *= oldt / t;  // update invD
        self.kappa *= self.delta;

        // if self.no_defer_trick
        // {
        //     self.mq *= self.kappa;
        //     self.kappa = 1.;
        // }
        return (status, self.tsq);  // g++-7 is ok
    }

