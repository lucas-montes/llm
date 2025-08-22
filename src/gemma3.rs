use crate::{modules::{gelu, Linear, Module}, tensor::Tensor};

pub struct FeedForward{
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
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


impl Module for FeedForward{
    type InitParams = InitParams;
    type ForwardParams = Tensor;

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
            linear3
        }
    }

    fn forward(&mut self, params: Self::ForwardParams) -> Tensor {
        let x1 = self.linear1.forward(params);
        let x2 = gelu(&x1, true).unwrap();
        let x3 = self.linear2.forward(x2);
        self.linear3.forward(x3)
    }
}
