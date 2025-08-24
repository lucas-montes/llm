use crate::{modules::Module, tensor::Tensor};

pub struct RMSNorm {
    eps: f32,
    scale: Tensor,
    shift: Option<Tensor>,
}

pub struct InitParams {
    bias: bool,
    emb_dim: usize,
    eps: f32,
}

impl InitParams {
    pub fn new(bias: bool, emb_dim: usize, eps: f32) -> Self {
        Self { bias, emb_dim, eps }
    }
}

impl Module for RMSNorm {
    type InitParams = InitParams;
    type ForwardParams<'a> = &'a Tensor;

    fn init(params: Self::InitParams) -> Self {
        Self {
            eps: params.eps,
            scale: Tensor::new(1, params.emb_dim),
            shift: params.bias.then(|| Tensor::new(1, params.emb_dim)),
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Tensor {
        let var = params.powf(2.0).mean();
        let norm = (&(&var + self.eps).rsqrt() * params).unwrap();
        let result = &norm * &(1.0 + &self.scale);
        match (result, self.shift.as_ref()) {
            (Ok(result), Some(shift)) => (&result + shift).unwrap(),
            (Ok(result), None) => result,
            (Err(e), _) => panic!("Error during forward pass: {}", e),
        }
    }
}
