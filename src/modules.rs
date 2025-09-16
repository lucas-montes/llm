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

pub fn gelu(input: &Tensor, approximate: bool) -> Result<Tensor, TensorError> {
    if approximate {
        let input_cube = (&(input * input)? * input)?;
        let inner = K_BETA * &(input + &(&input_cube * K_KAPPA))?;
        &(0.5 * input) * &(1.0 + &inner.tanh())
    } else {
        &(input * 0.5) * &(1.0 + &(input * K_ALPHA).erf())
    }
}

pub struct Mask {
    global: Tensor,
    local: Tensor,
}

impl Mask {
    pub fn new(seq_len: usize, sliding_window: isize) -> Self {
        let ones = Tensor::ones(&[seq_len, seq_len]);

        let global = ones.triu(1);

        let far_past = global.triu(sliding_window).transpose(0, 1);

        let local = global.mask(&far_past);
        Self { global, local }
    }

    pub fn global(&self) -> &Tensor {
        &self.global
    }

    pub fn local(&self) -> &Tensor {
        &self.local
    }
}

pub enum AttentionType {
    Sliding,
    Full,
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
}
