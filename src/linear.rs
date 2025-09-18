use crate::{modules::Module, tensor::{Tensor, TensorError}};

/// Applies an affine linear transformation to the incoming data: y=xAT+by=xA T+b.
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl From<InitParams> for Linear {
    fn from(params: InitParams) -> Self {
        Self::init(params)
    }
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
            weight: Tensor::rand(&[params.out_features, params.in_features], params.seed),
            bias: params
                .bias
                .then(|| Tensor::rand(&[1, params.out_features], params.seed)),
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Result<Tensor, TensorError> {
        //NOTE: test to have bias set to 0 so we always do the operation, we remove the branch
        let result = params.matmul(&self.weight.transpose(0, 1));
        match self.bias.as_ref() {
            Some(b) => &result? + b,
            None => result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let output = linear.forward(&input).unwrap();
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
        let output = linear.forward(&input).unwrap();
        assert_eq!(
            output.data(),
            &[
                3.5983946, 3.1537957, 1.8494523, 7.1967893, 6.130426, 3.098248
            ]
        );
    }
}
