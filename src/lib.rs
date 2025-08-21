use tensor::Tensor;

mod linear;
mod tensor;

trait Module {
    type InitParams;
    type ForwardParams;
    fn init(params: Self::InitParams) -> Self;
    // Forward pass through the module, this is the main computation
    fn forward(&mut self, params: Self::ForwardParams) -> Tensor;
}

