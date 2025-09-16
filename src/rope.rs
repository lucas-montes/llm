use crate::tensor::Tensor;

pub struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    pub fn new(head_dim: usize, theta_base: f32, context_length: usize) -> Self {
        assert!(head_dim % 2 == 0, "Embedding dimension must be even");

        // Compute the inverse frequencies
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .take(head_dim / 2)
            .map(|i| 1.0 / theta_base.powf(i as f32 / head_dim as f32))
            .collect();

        // vec![vec![0.0; head_dim]; context_length];

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
        let dims = x.shape().dims();
        let seq_len = dims[2];
        let head_dim = dims[3];

        // Extract cos and sin for current sequence
        let cos = self
            .cos
            .slice(&[0..seq_len, 0..head_dim])
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let sin = self
            .sin
            .slice(&[0..seq_len, 0..head_dim])
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        // Split into halves
        let x1 = x
            .slice(&[0..dims[0], 0..dims[1], 0..dims[2], 0..(head_dim / 2)])
            .unwrap();
        let x2 = x
            .slice(&[0..dims[0], 0..dims[1], 0..dims[2], (head_dim / 2)..head_dim])
            .unwrap();

        // Create rotated tensor: [-x2, x1]
        let neg_x2 = x2.neg();
        let rotated = Tensor::cat(&[&neg_x2, &x1], -1).unwrap();

        // Apply rotation: x * cos + rotated * sin
        let x_cos = (x * &cos).unwrap();
        let rotated_sin = (&rotated * &sin).unwrap();

        (&x_cos + &rotated_sin).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rope() {
        let rope = Rope::new(4, 10000.0, 3);

        let x = Tensor::from([[
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [2.0, 4.0, 6.0, 8.0],
                [10.0, 12.0, 14.0, 16.0],
                [18.0, 20.0, 22.0, 24.0],
            ],
        ]]);

        let result = rope.apply(&x);
        let expected = Tensor::from([[
            [
                [1.0000, 2.0000, 3.0000, 4.0000],
                [-3.1887855530, 5.9197015762, 7.9894704819, 8.0595989227],
                [-13.7475929260, 9.7580165863, 3.6060614586, 12.1975870132],
            ],
            [
                [2.0000000000, 4.0000000000, 6.0000000000, 8.0000000000],
                [-6.3775711060, 11.8394031525, 15.9789409637, 16.1191978455],
                [-27.4951858521, 19.5160331726, 7.2121229172, 24.3951740265],
            ],
        ]]);

        assert_eq!(result, expected);
    }

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
