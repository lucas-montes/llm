use crate::{modules::Module, tensor::Tensor};

use super::{
    feed_forward::FeedForward, grouped_query_attention::GroupedQueryAttention, rms_norm::RMSNorm,
    rope::Rope,
};

pub struct TransformerBlock {
    att: GroupedQueryAttention,
    ff: FeedForward,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    post_feedforward_layernorm: RMSNorm,
}

impl From<InitParams> for TransformerBlock {
    fn from(params: InitParams) -> Self {
        Self::init(params)
    }
}

pub struct InitParams {
    emb_dim: usize,
    n_heads: usize,
    n_kv_groups: usize,
    head_dim: Option<usize>,
    hidden_dim: usize,
    qk_norm: bool,
    query_pre_attn_scalar: Option<f32>,
    seed: Option<u64>,
    bias: bool,
}

impl InitParams {
    pub fn new(
        emb_dim: usize,
        n_heads: usize,
        n_kv_groups: usize,
        head_dim: Option<usize>,
        hidden_dim: usize,
        qk_norm: bool,
        query_pre_attn_scalar: Option<f32>,
        seed: Option<u64>,
        bias: bool,
    ) -> Self {
        Self {
            emb_dim,
            n_heads,
            n_kv_groups,
            head_dim,
            hidden_dim,
            qk_norm,
            query_pre_attn_scalar,
            seed,
            bias,
        }
    }
}

pub struct ForwardParams<'a> {
    x: &'a Tensor,
    mask: &'a Tensor,
    rope: &'a Rope,
}

impl<'a> ForwardParams<'a> {
    pub fn new(x: &'a Tensor, mask: &'a Tensor, rope: &'a Rope) -> Self {
        Self { x, mask, rope }
    }
}

impl Module for TransformerBlock {
    type InitParams = InitParams;
    type ForwardParams<'a> = ForwardParams<'a>;

    fn init(params: Self::InitParams) -> Self {
        let att = super::grouped_query_attention::InitParams::new(
            params.emb_dim,
            params.n_heads,
            params.n_kv_groups,
            params.head_dim,
            params.qk_norm,
            params.query_pre_attn_scalar,
            params.seed,
        )
        .into();

        let ff = super::feed_forward::InitParams::new(
            params.bias,
            params.emb_dim,
            params.hidden_dim,
            params.seed,
        )
        .into();

        let norm = || super::rms_norm::InitParams::new(false, params.emb_dim, 1e-6);

        Self {
            att,
            ff,
            input_layernorm: norm().into(),
            post_attention_layernorm: norm().into(),
            pre_feedforward_layernorm: norm().into(),
            post_feedforward_layernorm: norm().into(),
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Tensor {

        let input_layernorm = self.input_layernorm.forward(params.x);

        let attn_params = super::grouped_query_attention::ForwardParams::new(
            &input_layernorm,
            params.mask,
            params.rope,
        );
        let mut x_attn = self.att.forward(attn_params);
        x_attn = self.post_attention_layernorm.forward(&x_attn);

        let x = (&x_attn + params.x).unwrap();

        let mut x_ffn = self.pre_feedforward_layernorm.forward(&x);
        x_ffn = self.ff.forward(&x_ffn);
        x_ffn = self.post_feedforward_layernorm.forward(&x_ffn);
        (&x_ffn + &x).unwrap()
    }
}
