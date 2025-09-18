use crate::{
    modules::Module,
    tensor::{Tensor, TensorError},
};

pub struct RMSNorm {
    eps: f32,
    scale: Tensor,
    shift: Option<Tensor>,
}

impl From<InitParams> for RMSNorm {
    fn from(params: InitParams) -> Self {
        Self::init(params)
    }
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
            scale: Tensor::zero(&[1, params.emb_dim]),
            shift: params.bias.then(|| Tensor::zero(&[1, params.emb_dim])),
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Result<Tensor, TensorError> {
        let var = params.powf(2.0).mean(None, false)?;
        let norm = (&(&var + self.eps).rsqrt() * params)?;
        let result = &norm * &(1.0 + &self.scale);
        match self.shift.as_ref() {
            Some(shift) => &result? + shift,
            None => result,
        }
    }
}
