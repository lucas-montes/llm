use crate::{modules::Module, tensor::{Tensor, TensorError}};

// https://github.com/pytorch/pytorch/blob/886699bc5c23105f6105d329f6ff6c0ada7b473c/torch/csrc/api/include/torch/nn/functional/embedding.h#L22
pub struct Embedding {
    weight: Tensor, // Shape: (num_embeddings, embedding_dim)
    num_embeddings: usize,
    embedding_dim: usize,
}

impl From<InitParams> for Embedding {
    fn from(params: InitParams) -> Self {
        Self::init(params)
    }
}

pub struct InitParams {
    num_embeddings: usize, // vocab_size
    embedding_dim: usize,  // emb_dim
    seed: Option<u64>,
}

impl InitParams {
    pub fn new(num_embeddings: usize, embedding_dim: usize, seed: Option<u64>) -> Self {
        Self {
            num_embeddings,
            embedding_dim,
            seed,
        }
    }
}

impl Module for Embedding {
    type InitParams = InitParams;
    type ForwardParams<'a> = &'a [usize];

    fn init(params: Self::InitParams) -> Self {
        let weight = Tensor::rand(&[params.num_embeddings, params.embedding_dim], params.seed);

        Self {
            weight,
            num_embeddings: params.num_embeddings,
            embedding_dim: params.embedding_dim,
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Result<Tensor, TensorError> {
        // Calculate output size: each token becomes embedding_dim numbers
        let num_tokens = params.len();
        let total_elements = num_tokens * self.embedding_dim;
        let mut output_data = Vec::with_capacity(total_elements);
        let data = self.weight.data();

        // For each token ID, lookup its embedding vector
        for &token_id in params {
            assert!(
                token_id < self.num_embeddings,
                "Token ID {} >= vocab_size {}",
                token_id,
                self.num_embeddings
            );

            // Calculate where this token's embedding starts in the weight matrix
            // weight matrix is stored as flat array: [token0_dim0, token0_dim1, ..., token1_dim0, token1_dim1, ...]
            let start_idx = token_id * self.embedding_dim;
            let end_idx = start_idx + self.embedding_dim;

            // Copy this token's embedding vector to output
            output_data.extend_from_slice(&data[start_idx..end_idx]);
        }
        // ALWAYS return with batch dimension: [1, seq_len, emb_dim] like pytorch
        Ok(Tensor::new(&[1, num_tokens, self.embedding_dim], output_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding() {
        let vocab_size = 10;
        let emb_dim = 4;
        let seed = Some(42);

        let mut embedding: Embedding = InitParams::new(vocab_size, emb_dim, seed).into();

        let input_tokens = vec![0, 1, 2, 3, 4];
        let output = embedding.forward(&input_tokens).unwrap();

        let expected = Tensor::new(
            &[1, 5, emb_dim],
            vec![
                0.1334097385,
                0.5265573859,
                0.2487383485,
                0.5427252054,
                0.8684265614,
                0.6364650726,
                0.9900846481,
                0.4059017301,
                0.9690188766,
                0.0343427658,
                0.6174240708,
                0.4149568081,
                0.3477854729,
                0.7374243736,
                0.1857714057,
                0.8492515683,
                0.6293413043,
                0.1312788725,
                0.4794040918,
                0.0032520890,
            ],
        );

        assert_eq!(output, expected);
    }
}
