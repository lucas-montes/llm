use crate::tensor::Tensor;

pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    pub fn new(head_dim: usize, theta_base: f32, context_length: usize)->Self{
        assert!(head_dim % 2 == 0, "Embedding dimension must be even");

    // Compute the inverse frequencies
    let inv_freq: Vec<_> = (0..head_dim)
        .step_by(2)
        .take(head_dim / 2)
        .map(|i| 1.0 / theta_base.powf(i as f32 / head_dim as f32))
        .collect();

    vec![vec![0.0; head_dim]; context_length];

    //TODO: try to iter over the inv_freq without collecting. Loop over context_length and set the values

    let mut cos = Vec::with_capacity(context_length);
    let mut sin = Vec::with_capacity(context_length);

    for i in 0..context_length {
        let mut cos_temp = vec![0.0; head_dim / 2];
        let mut sin_temp = vec![0.0; head_dim / 2];

        for (h, &freq) in inv_freq.iter().enumerate() {
            let angle = i as f32 * freq;
            cos_temp[h] = angle.cos();
            sin_temp[h] = angle.sin();
        }

        cos_temp.extend_from_within(..);
        sin_temp.extend_from_within(..);

        cos.push(cos_temp);
        sin.push(sin_temp);
    }

    Self {
        cos: Tensor::from(cos),
        sin: Tensor::from(sin),
    }
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rope() {
        let result = Rope::new(10, 10000.0, 8);

        let expected_cos = Tensor::from([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [
                0.5403023, 0.9874668, 0.9996845, 0.9999921, 0.9999998, 0.5403023, 0.9874668,
                0.9996845, 0.9999921, 0.9999998,
            ],
            [
                -0.41614684,
                0.9501815,
                0.99873835,
                0.9999683,
                0.9999992,
                -0.41614684,
                0.9501815,
                0.99873835,
                0.9999683,
                0.9999992,
            ],
            [
                -0.9899925, 0.8890786, 0.99716204, 0.99992865, 0.9999982, -0.9899925, 0.8890786,
                0.99716204, 0.99992865, 0.9999982,
            ],
            [
                -0.6536436, 0.8056898, 0.9949566, 0.9998732, 0.99999684, -0.6536436, 0.8056898,
                0.9949566, 0.9998732, 0.99999684,
            ],
            [
                0.2836622, 0.7021052, 0.9921234, 0.9998019, 0.99999505, 0.2836622, 0.7021052,
                0.9921234, 0.9998019, 0.99999505,
            ],
            [
                0.96017027, 0.58092153, 0.98866427, 0.99971473, 0.99999285, 0.96017027, 0.58092153,
                0.98866427, 0.99971473, 0.99999285,
            ],
            [
                0.75390226, 0.4451763, 0.98458135, 0.99961174, 0.9999902, 0.75390226, 0.4451763,
                0.98458135, 0.99961174, 0.9999902,
            ],
        ]);
        let expected_sin = Tensor::from([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.84147096,
                0.15782663,
                0.025116222,
                0.00398106,
                0.00063095725,
                0.84147096,
                0.15782663,
                0.025116222,
                0.00398106,
                0.00063095725,
            ],
            [
                0.9092974,
                0.31169716,
                0.050216597,
                0.007962057,
                0.0012619143,
                0.9092974,
                0.31169716,
                0.050216597,
                0.007962057,
                0.0012619143,
            ],
            [
                0.14112,
                0.45775455,
                0.075285286,
                0.011942928,
                0.0018928708,
                0.14112,
                0.45775455,
                0.075285286,
                0.011942928,
                0.0018928708,
            ],
            [
                -0.7568025,
                0.5923377,
                0.10030648,
                0.01592361,
                0.0025238264,
                -0.7568025,
                0.5923377,
                0.10030648,
                0.01592361,
                0.0025238264,
            ],
            [
                -0.9589243,
                0.7120732,
                0.12526439,
                0.019904038,
                0.0031547814,
                -0.9589243,
                0.7120732,
                0.12526439,
                0.019904038,
                0.0031547814,
            ],
            [
                -0.2794155,
                0.81395954,
                0.15014327,
                0.023884153,
                0.0037857348,
                -0.2794155,
                0.81395954,
                0.15014327,
                0.023884153,
                0.0037857348,
            ],
            [
                0.6569866,
                0.89544296,
                0.17492741,
                0.027863888,
                0.004416687,
                0.6569866,
                0.89544296,
                0.17492741,
                0.027863888,
                0.004416687,
            ],
        ]);
        assert_eq!(result.cos, expected_cos);
        assert_eq!(result.sin, expected_sin);
    }
}
