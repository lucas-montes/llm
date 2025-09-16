use crate::{
    embedding::{self, Embedding},
    linear::{self, Linear},
    modules::{AttentionType, Mask, Module},
    rms_norm::{self, RMSNorm},
    rope::Rope,
    tensor::Tensor,
    transformer_block::TransformerBlock,
};

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
            vocab_size: 262_144,
            context_length: 32_768,
            emb_dim: 640,
            n_heads: 4,
            n_layers: 18,
            hidden_dim: 2048,
            head_dim: 256,
            qk_norm: true,
            n_kv_groups: 1,
            rope_local_base: 10_000.0,
            rope_base: 1_000_000.0,
            sliding_window: 512,
            layer_types: vec![
                AttentionType::Sliding,    // 0
                AttentionType::Sliding,    // 1
                AttentionType::Sliding,    // 2
                AttentionType::Sliding,    // 3
                AttentionType::Sliding,    // 4
                AttentionType::Full,       // 5
                AttentionType::Sliding,    // 6
                AttentionType::Sliding,    // 7
                AttentionType::Sliding,    // 8
                AttentionType::Sliding,    // 9
                AttentionType::Sliding,    // 10
                AttentionType::Full,       // 11
                AttentionType::Sliding,    // 12
                AttentionType::Sliding,    // 13
                AttentionType::Sliding,    // 14
                AttentionType::Sliding,    // 15
                AttentionType::Sliding,    // 16
                AttentionType::Full,       // 17
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

pub struct Gemma3 {
    token_embeddings: Embedding,
    final_norm: RMSNorm,
    out_head: Linear,
    params: InitParams,
}

impl From<InitParams> for Gemma3 {
    fn from(params: InitParams) -> Self {
        Self::init(params)
    }
}

impl Module for Gemma3 {
    type InitParams = InitParams;
    type ForwardParams<'a> = &'a [usize];

    fn init(params: Self::InitParams) -> Self {
        // Validate layer_types length
        assert_eq!(params.layer_types.len(), params.n_layers);

        let token_embeddings = embedding::InitParams::new(params.vocab_size, params.emb_dim, params.seed).into();

        // Initialize final norm and output head
        let final_norm = rms_norm::InitParams::new(false, params.emb_dim, 1e-6).into();
        let out_head =
            linear::InitParams::new(false, params.emb_dim, params.vocab_size, params.seed).into();

        Self {
            token_embeddings,
            final_norm,
            out_head,
            params,
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Tensor {
        let seq_len = params.len();

        let mut x = &self.token_embeddings.forward(params) * (self.params.emb_dim as f32).powf(0.5);
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
                self.params.bias,
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

        let masks = Mask::new(seq_len, self.params.sliding_window as isize);

        for layer_type in &self.params.layer_types {
            let (rope, mask) = match layer_type {
                AttentionType::Sliding => (&local_rope, masks.local()),
                AttentionType::Full => (&global_rope, masks.global()),
            };
            let mut block: TransformerBlock = create_block().into();
            let block_params = crate::transformer_block::ForwardParams::new(&x, mask, rope);
            x = block.forward(block_params);
        }

        x = self.final_norm.forward(&x);
        self.out_head.forward(&x)
    }
}
