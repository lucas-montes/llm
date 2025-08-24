use crate::{
    modules::{Linear, Module},
    tensor::Tensor,
};

use super::{rms_norm::RMSNorm, rope::Rope};

pub struct GroupedQueryAttention {
    num_heads: usize,
    num_kv_groups: usize,
    group_size: usize,
    head_dim: usize,
    d_out: usize,
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    out_proj: Linear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    scaling: f32,
}

pub struct InitParams {
    d_in: usize,
    num_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    qk_norm: bool,
    query_pre_attn_scalar: Option<f32>,
    seed: Option<u64>,
}

impl InitParams {
    pub fn new(
        d_in: usize,
        num_heads: usize,
        num_kv_groups: usize,
        head_dim: Option<usize>,
        qk_norm: bool,
        query_pre_attn_scalar: Option<f32>,
        seed: Option<u64>,
    ) -> Self {
        assert_eq!(
            num_heads % num_kv_groups,
            0,
            "num_heads must be divisible by num_kv_groups"
        );
        let head_dim = head_dim.unwrap_or_else(|| {
            assert_eq!(
                d_in % num_heads,
                0,
                "d_in must be divisible by num_heads when head_dim is not specified"
            );
            d_in / num_heads
        });

        Self {
            d_in,
            num_heads,
            num_kv_groups,
            head_dim,
            qk_norm,
            query_pre_attn_scalar,
            seed,
        }
    }
}

pub struct ForwardParams {
    x: Tensor,
    mask: Option<Tensor>,
    rope: Rope,
}

impl Module for GroupedQueryAttention {
    type InitParams = InitParams;
    type ForwardParams<'a> = ForwardParams;

    fn init(params: Self::InitParams) -> Self {
        let d_out = params.num_heads * params.head_dim;

        let new_linear = |in_features, out_features| {
            Linear::init(<Linear as Module>::InitParams::new(
                false,
                in_features,
                out_features,
                params.seed,
            ))
        };

        let w_query = new_linear(params.d_in, d_out);

        let w_key = new_linear(params.d_in, params.num_kv_groups * params.head_dim);
        let w_value = new_linear(params.d_in, params.num_kv_groups * params.head_dim);
        let out_proj = new_linear(d_out, params.d_in);

        let new_rms = || {
            RMSNorm::init(<RMSNorm as Module>::InitParams::new(
                false,
                params.head_dim,
                1e-6,
            ))
        };

        let q_norm = params.qk_norm.then(new_rms);
        let k_norm = params.qk_norm.then(new_rms);

        let scaling = params
            .query_pre_attn_scalar
            .unwrap_or((params.head_dim as f32).powf(-0.5));

        Self {
            num_heads: params.num_heads,
            num_kv_groups: params.num_kv_groups,
            group_size: params.num_heads / params.num_kv_groups,

            head_dim: params.head_dim,
            d_out,
            w_query,
            w_key,
            w_value,
            out_proj,
            q_norm,
            k_norm,

            scaling,
        }
    }

    fn forward<'a>(&mut self, params: Self::ForwardParams<'a>) -> Tensor {
        let queries = self.w_query.forward(&params.x);
        let keys = self.w_key.forward(&params.x);
        let values = self.w_value.forward(&params.x);
        todo!()
    }
}
