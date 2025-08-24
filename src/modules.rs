use crate::tensor::{Tensor, TensorError};

pub trait Module {
    type InitParams;
    type ForwardParams<'a>;
    fn init(params: Self::InitParams) -> Self;
    // Forward pass through the module, this is the main computation
    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Tensor;
}

const M_2_SQRTPI: f32 = 1.12837916709551257390;
const M_SQRT1_2: f32 = 0.70710678118654752440;
const M_SQRT2: f32 = 1.41421356237309504880;
const K_BETA: f32 = M_SQRT2 * M_2_SQRTPI * 0.5;
const K_KAPPA: f32 = 0.044715;
const K_ALPHA: f32 = M_SQRT1_2;

/// Applies an affine linear transformation to the incoming data: y=xAT+by=xA T+b.
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

pub struct InitParams {
    bias: bool,
    in_features: usize,
    out_features: usize,
    seed: Option<u64>,
}

impl InitParams {
    pub fn new(bias: bool, in_features: usize, out_features: usize, seed: Option<u64>) -> Self {
        Self {
            bias,
            in_features,
            out_features,
            seed,
        }
    }
}

impl Module for Linear {
    type InitParams = InitParams;
    type ForwardParams<'a> = &'a Tensor;

    fn init(params: Self::InitParams) -> Self {
        Self {
            weight: Tensor::rand(params.out_features, params.in_features, params.seed),
            bias: params
                .bias
                .then(|| Tensor::rand(1, params.out_features, params.seed)),
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Tensor {
        let result = params.matmul(&self.weight.transpose());
        let result = match (result, self.bias.as_ref()) {
            (Ok(result), Some(b)) => &result + b,
            (Ok(result), None) => Ok(result),
            (Err(e), _) => panic!("Error during forward pass: {}", e),
        };
        result.unwrap()
    }
}

pub fn gelu(input: &Tensor, approximate: bool) -> Result<Tensor, TensorError> {
    if approximate {
        let input_cube = (&(input * input)? * input)?;
        let inner = K_BETA * &(input + &(&input_cube * K_KAPPA))?;
        &(0.5 * input) * &(1.0 + &inner.tanh())
    } else {
        &(input * 0.5) * &(1.0 + &(input * K_ALPHA).erf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_tanh() {
        let input = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let output = gelu(&input, true).unwrap();
        assert_eq!(output.data(), &[0.841192, 1.9545977, 2.9963627, 3.99993]);
    }

    #[test]
    fn test_gelu_erf() {
        let input = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let output = gelu(&input, false).unwrap();
        assert_eq!(output.data(), &[0.8413447, 1.9544997, 2.9959502, 4.0]);
    }

    #[test]
    fn test_linear() {
        let init_params = InitParams {
            bias: false,
            in_features: 2,
            out_features: 3,
            seed: Some(1),
        };
        let mut linear = Linear::init(init_params);
        let input = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let output = linear.forward(&input);
        assert_eq!(
            output.data(),
            &[
                2.773442, 2.179551, 1.0523727, 6.3718367, 5.1561813, 2.3011684
            ]
        );
    }

    #[test]
    fn test_linear_with_bias() {
        let init_params = InitParams {
            bias: true,
            in_features: 2,
            out_features: 3,
            seed: Some(1),
        };
        let mut linear = Linear::init(init_params);
        let input = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let output = linear.forward(&input);
        assert_eq!(
            output.data(),
            &[
                3.5983946, 3.1537957, 1.8494523, 7.1967893, 6.130426, 3.098248
            ]
        );
    }
}
