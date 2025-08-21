use crate::tensor::Tensor;

trait Module {
    type InitParams;
    type ForwardParams;
    fn init(params: Self::InitParams) -> Self;
    // Forward pass through the module, this is the main computation
    fn forward(&mut self, params: Self::ForwardParams) -> Tensor;
}

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

impl Module for Linear {
    type InitParams = InitParams;
    type ForwardParams = Tensor;

    fn init(params: Self::InitParams) -> Self {
        //TODO: we probably want to pass the seed
        Self {
            weight: Tensor::rand(params.out_features, params.in_features, params.seed),
            bias: params
                .bias
                .then(|| Tensor::rand(1, params.out_features, params.seed)),
        }
    }

    fn forward(&mut self, params: Self::ForwardParams) -> Tensor {
        println!("Forward pass through Linear module with input shape: {:?}", self.bias.as_ref());
        let result = params.matmul(&self.weight.transpose());
        let result = match (result, self.bias.as_ref()) {
            (Ok(result), Some(b)) => &result + b,
            (Ok(result), None) => Ok(result),
            (Err(e), _) => panic!("Error during forward pass: {}", e),
        };
        result.unwrap()
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
        let output = linear.forward(input);
        assert_eq!(output.data(), &[2.773442, 2.179551, 1.0523727, 6.3718367, 5.1561813, 2.3011684]);

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
        let output = linear.forward(input);
        assert_eq!(output.data(), &[3.5983946, 3.1537957, 1.8494523, 7.1967893, 6.130426, 3.098248]);

    }
}
