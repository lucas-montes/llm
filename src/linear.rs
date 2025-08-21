use crate::{Module, tensor::Tensor};

/// Applies an affine linear transformation to the incoming data: y=xAT+by=xA T+b.
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

pub struct InitParams {
    bias: bool,
    in_features: usize,
    out_features: usize,
}

impl Module for Linear {
    type InitParams = InitParams;
    type ForwardParams = Tensor;

    fn init(params: Self::InitParams) -> Self {
       todo!()
    }

    fn forward(&mut self, params: Self::ForwardParams) -> Tensor {
        todo!()
    }
}
