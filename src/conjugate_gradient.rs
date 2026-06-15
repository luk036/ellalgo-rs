use crate::arr::Arr;

pub fn conjugate_gradient(
    a: &Arr,
    b: &Arr,
    x0: Option<&Arr>,
    tol: f64,
    max_iter: usize,
) -> Result<Arr, &'static str> {
    let n = b.len();
    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Arr::new(n),
    };

    let mut residual = b - &a.dot_mv(&x);
    let mut direction = residual.clone();
    let mut residual_norm_sq = residual.dot(&residual);

    for _iter in 0..max_iter {
        let a_dir = a.dot_mv(&direction);
        let dir_dot_a_dir = direction.dot(&a_dir);

        if dir_dot_a_dir == 0.0 {
            return Err("Conj Grad did not converge");
        }

        let step_size = residual_norm_sq / dir_dot_a_dir;
        // x += step_size * direction
        for i in 0..n {
            x[i] += step_size * direction[i];
        }
        // residual -= step_size * a_dir
        for i in 0..n {
            residual[i] -= step_size * a_dir[i];
        }

        let residual_norm_sq_new = residual.dot(&residual);

        if residual_norm_sq_new.sqrt() < tol {
            return Ok(x);
        }

        let improvement_ratio = residual_norm_sq_new / residual_norm_sq;
        // direction = residual + improvement_ratio * direction
        for i in 0..n {
            direction[i] = residual[i] + improvement_ratio * direction[i];
        }
        residual_norm_sq = residual_norm_sq_new;
    }

    Err("Conj Grad did not converge after max iterations")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    fn a_matrix() -> Arr {
        Arr::with_data(vec![4.0, 1.0, 1.0, 3.0], 2, 2)
    }

    fn b_vector() -> Arr {
        Arr::from(vec![1.0, 2.0])
    }

    #[test]
    fn test_conjugate_gradient_simple() {
        let a = a_matrix();
        let b = b_vector();
        let x = conjugate_gradient(&a, &b, None, 1e-5, 1000).unwrap();
        assert_approx_eq!(x[0], 0.0909091, 1e-5);
        assert_approx_eq!(x[1], 0.6363636, 1e-5);
    }

    #[test]
    fn test_conjugate_gradient_with_initial_guess() {
        let a = a_matrix();
        let b = b_vector();
        let x0 = Arr::from(vec![1.0, 1.0]);
        let x = conjugate_gradient(&a, &b, Some(&x0), 1e-5, 1000).unwrap();
        assert_approx_eq!(x[0], 0.0909091, 1e-5);
        assert_approx_eq!(x[1], 0.6363636, 1e-5);
    }

    #[test]
    fn test_conjugate_gradient_non_convergence() {
        let a = Arr::with_data(vec![0.0, 0.0, 0.0, 0.0], 2, 2);
        let b = Arr::from(vec![1.0, 1.0]);
        let result = conjugate_gradient(&a, &b, None, 1e-5, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_conjugate_gradient_tolerance() {
        let a = a_matrix();
        let b = b_vector();
        let tol = 1e-10;
        let x = conjugate_gradient(&a, &b, None, tol, 1000).unwrap();
        let residual = &b - &a.dot_mv(&x);
        let err = residual.dot(&residual).sqrt();
        assert!(err < tol);
    }
}
