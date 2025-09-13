use crate::{
    linear::{self, Linear},
    modules::{AttentionType, Mask, Module},
    rms_norm::{self, RMSNorm},
    tensor::Tensor,
    transformer_block::TransformerBlock,
};

use super::rope::Rope;

pub struct Gemma3 {
    // token_embeddings
    // blocks: Vec<TransformerBlock>,
    // blocks: Vec<crate::transformer_block::InitParams>,
    final_norm: RMSNorm,
    out_head: Linear,
    // emb_dim: usize,
    params: InitParams,
}

impl From<InitParams> for Gemma3 {
    fn from(params: InitParams) -> Self {
        Self::init(params)
    }
}

pub struct InitParams {
    vocab_size: usize,
    context_length: usize,
    emb_dim: usize,
    n_heads: usize,
    n_layers: usize,
    hidden_dim: usize,
    head_dim: usize,
    qk_norm: bool,
    n_kv_groups: usize,
    rope_local_base: f32,
    rope_base: f32,
    sliding_window: usize,
    layer_types: Vec<AttentionType>,
    query_pre_attn_scalar: Option<f32>,
    seed: Option<u64>,
    bias: bool,
}

impl InitParams {
    pub fn gemma3_270m() -> Self {
        Self {
            vocab_size: 44,
            context_length: 8,
            emb_dim: 4,
            n_heads: 4,
            n_layers: 18,
            hidden_dim: 8,
            head_dim: 8,
            qk_norm: true,
            n_kv_groups: 1,
            rope_local_base: 10_000.0,
            rope_base: 1_000_000.0,
            sliding_window: 512,
            layer_types: vec![
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Full,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Full,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Full,
            ],
            query_pre_attn_scalar: Some(256.0),
            seed: None,
            bias: false,
        }
    }

    pub fn new(
        vocab_size: usize,
        context_length: usize,
        emb_dim: usize,
        n_heads: usize,
        n_layers: usize,
        hidden_dim: usize,
        head_dim: usize,
        qk_norm: bool,
        n_kv_groups: usize,
        rope_local_base: f32,
        rope_base: f32,
        sliding_window: usize,
        layer_types: Vec<AttentionType>,
        query_pre_attn_scalar: Option<f32>,
        seed: Option<u64>,
        bias: bool,
    ) -> Self {
        Self {
            vocab_size,
            context_length,
            emb_dim,
            n_heads,
            n_layers,
            hidden_dim,
            head_dim,
            qk_norm,
            n_kv_groups,
            rope_local_base,
            rope_base,
            sliding_window,
            layer_types,
            query_pre_attn_scalar,
            seed,
            bias,
        }
    }
}

impl Module for Gemma3 {
    type InitParams = InitParams;
    type ForwardParams<'a> = Tensor;

    fn init(params: Self::InitParams) -> Self {
        // Validate layer_types length
        assert_eq!(params.layer_types.len(), params.n_layers);

        // Initialize embedding (random for now)
        let tok_emb = Tensor::rand(&[params.vocab_size, params.emb_dim], None);

        // let create_block = || crate::transformer_block::InitParams::new(
        //         params.emb_dim,
        //         params.n_heads,
        //         params.n_kv_groups,
        //         Some(params.head_dim),//TODO: check why this is an option
        //         params.hidden_dim,
        //         params.qk_norm,
        //         params.query_pre_attn_scalar,
        //         params.seed,
        //         params.bias
        //         //TODO: check how to pass rope according to the type of attention
        //     );

        // // Initialize transformer blocks
        // let  blocks = (0..params.n_layers).map(|_|create_block()).collect();

        // Initialize final norm and output head
        let final_norm = rms_norm::InitParams::new(false, params.emb_dim, 1e-6).into();
        let out_head =
            linear::InitParams::new(false, params.emb_dim, params.vocab_size, params.seed).into();

        // Compute RoPE parameters
        // let local_rope = Rope::new(
        //     params.head_dim,
        //     params.rope_local_base,
        //     params.context_length,
        // );
        // let global_rope = Rope::new(params.head_dim, params.rope_base, params.context_length);

        Self {
            // tok_emb,
            // blocks,
            final_norm,
            out_head,
            // emb_dim: params.emb_dim,
            params,
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Tensor {
        let seq_len = params.shape().dims()[1];//TODO: check if this is correct

        let mut x = params; // Input tensor of shape (batch_size, seq_length, emb_dim)
        let create_block = || {
            crate::transformer_block::InitParams::new(
                self.params.emb_dim,
                self.params.n_heads,
                self.params.n_kv_groups,
                Some(self.params.head_dim), //NOTE: for GroupedQueryAttention the head_dim could be none
                self.params.hidden_dim,
                self.params.qk_norm,
                self.params.query_pre_attn_scalar,
                self.params.seed,
                self.params.bias, //TODO: check how to pass rope according to the type of attention
            )
        };

        let local_rope = Rope::new(
            self.params.head_dim,
            self.params.rope_local_base,
            self.params.context_length,
        );
        let global_rope = Rope::new(
            self.params.head_dim,
            self.params.rope_base,
            self.params.context_length,
        );

        let masks = Mask::new(seq_len);

        for layer_type in &self.params.layer_types {
            let (rope, mask) = match layer_type {
                AttentionType::Sliding => (&local_rope, masks.local()),
                AttentionType::Full => (&global_rope, masks.global()),
            };
            let mut block: TransformerBlock = create_block().into();
            let block_params = crate::transformer_block::ForwardParams::new(
                &x,
                mask,
                rope,
            );
           x = block.forward(block_params);
        }

        x = self.final_norm.forward(&x);
        self.out_head.forward(&x)
    }
}
