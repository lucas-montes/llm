use crate::{
    linear::Linear,
    modules::{Module, gelu},
    tensor::{Tensor, TensorError},
};

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
}

impl From<InitParams> for FeedForward {
    fn from(params: InitParams) -> Self {
        Self::init(params)
    }
}

pub struct InitParams {
    bias: bool,
    emb_dim: usize,
    hidden_dim: usize,
    seed: Option<u64>,
}

impl InitParams {
    pub fn new(bias: bool, emb_dim: usize, hidden_dim: usize, seed: Option<u64>) -> Self {
        Self {
            bias,
            emb_dim,
            hidden_dim,
            seed,
        }
    }
}

impl Module for FeedForward {
    type InitParams = InitParams;
    type ForwardParams<'a> = &'a Tensor;

    fn init(params: Self::InitParams) -> Self {
        let linear1 = Linear::init(<Linear as Module>::InitParams::new(
            params.bias,
            params.emb_dim,
            params.hidden_dim,
            params.seed,
        ));
        let linear2 = Linear::init(<Linear as Module>::InitParams::new(
            params.bias,
            params.emb_dim,
            params.hidden_dim,
            params.seed,
        ));
        let linear3 = Linear::init(<Linear as Module>::InitParams::new(
            params.bias,
            params.hidden_dim,
            params.emb_dim,
            params.seed,
        ));
        Self {
            linear1,
            linear2,
            linear3,
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Result<Tensor, TensorError> {
        // let x1 = self.linear1.forward(params)?;
        // let x2 = self.linear2.forward(params)?;

        let (x1, x2) = rayon::join(
            || self.linear1.forward(params),
            || self.linear2.forward(params),
        );

        let x3 = (&gelu(&x1?, true)? * &x2?)?;

        self.linear3.forward(&x3)
    }
}
