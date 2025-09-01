use std::{
    arch::x86_64::{__m128, _mm_load_ps, _mm_rsqrt_ps},
    fmt,
    ops::{Add, Div, Mul, Sub},
};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::erf::erf;

#[derive(Debug)]
pub enum TensorError {
    DimensionMismatch {
        expected: (usize, usize),
        found: (usize, usize),
    },
    InvalidMatrixMultiplication {
        left_cols: usize,
        right_rows: usize,
    },
    InvalidDataLength {
        expected: usize,
        found: usize,
    },
    DivisionByZero,
    IndexOutOfBounds {
        row: usize,
        col: usize,
        shape: Shape,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, found {:?}",
                    expected, found
                )
            }
            TensorError::InvalidMatrixMultiplication {
                left_cols,
                right_rows,
            } => {
                write!(
                    f,
                    "Invalid matrix multiplication: left cols {} != right rows {}",
                    left_cols, right_rows
                )
            }
            TensorError::InvalidDataLength { expected, found } => {
                write!(
                    f,
                    "Invalid data length: expected {}, found {}",
                    expected, found
                )
            }
            TensorError::DivisionByZero => {
                write!(f, "Division by zero")
            }
            TensorError::IndexOutOfBounds { row, col, shape } => {
                write!(
                    f,
                    "Index ({}, {}) out of bounds for shape {:?}",
                    row, col, shape
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    cols: usize,
    rows: usize,
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    fn size(&self) -> usize {
        self.rows * self.cols
    }

    fn broadcast(&self, other: &Shape) -> Result<Shape, TensorError> {
        // Treat shapes as 2D, but allow rows=0 or cols=0 to represent 1D or scalar tensors
        let rows_a = self.rows;
        let cols_a = self.cols;
        let rows_b = other.rows;
        let cols_b = other.cols;

        // Compute broadcasted dimensions
        let out_rows = match (rows_a, rows_b) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            (0, b) => b, // Treat 0 as 1 for broadcasting (e.g., empty tensor)
            (a, 0) => a,
            _ => {
                return Err(TensorError::DimensionMismatch {
                    expected: (rows_a, cols_a),
                    found: (rows_b, cols_b),
                });
            }
        };

        let out_cols = match (cols_a, cols_b) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            (0, b) => b,
            (a, 0) => a,
            _ => {
                return Err(TensorError::DimensionMismatch {
                    expected: (rows_a, cols_a),
                    found: (rows_b, cols_b),
                });
            }
        };

        Ok(Shape {
            rows: out_rows,
            cols: out_cols,
        })
    }
}

pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Default::default(); rows * cols],
            shape: Shape { rows, cols },
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![1.0; rows * cols],
            shape: Shape { rows, cols },
        }
    }

    pub fn triu(&self, k: isize) -> Tensor {
        let rows = self.shape.rows;
        let cols = self.shape.cols;
        let mut data = self.data.clone();

        for i in 0..rows {
            for j in 0..cols {
                // Only keep elements where j - i >= k
                if (j as isize) - (i as isize) < k {
                    data[i * cols + j] = 0.0;
                }
            }
        }

        Tensor {
            data,
            shape: Shape { rows, cols },
        }
    }

    pub fn repeat_interleave(&self, repeats: usize, dim: usize) -> Tensor {
        let rows = self.shape.rows;
        let cols = self.shape.cols;
        let mut data = Vec::new();

        match dim {
            0 => {
                // repeat rows
                for i in 0..rows {
                    for _ in 0..repeats {
                        for j in 0..cols {
                            data.push(self.data[i * cols + j]);
                        }
                    }
                }
                Tensor {
                    data,
                    shape: Shape {
                        rows: rows * repeats,
                        cols,
                    },
                }
            }
            1 => {
                // repeat columns
                for i in 0..rows {
                    for j in 0..cols {
                        for _ in 0..repeats {
                            data.push(self.data[i * cols + j]);
                        }
                    }
                }
                Tensor {
                    data,
                    shape: Shape {
                        rows,
                        cols: cols * repeats,
                    },
                }
            }
            _ => panic!("Invalid dim for repeat_interleave"),
        }
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor, TensorError> {
        if self.shape != mask.shape {
            return Err(TensorError::DimensionMismatch {
                expected: (self.shape.rows, self.shape.cols),
                found: (mask.shape.rows, mask.shape.cols),
            });
        }

        let data = self
            .data
            .iter()
            .zip(mask.data.iter())
            .map(|(x, m)| if *m != 0.0 { *x } else { value })
            .collect();

        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn rand(rows: usize, cols: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let mut data = vec![Default::default(); rows * cols];
        data.fill_with(|| rng.random());

        Self {
            data,
            shape: Shape { rows, cols },
        }
    }

    pub fn powf(&self, exp: f32) -> Tensor {
        let data = self.data.iter().map(|x| x.powf(exp)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn mean(&self) -> Tensor {
        let cols = self.shape.cols as f32;

        let data = (0..self.shape.rows)
            .map(|i| {
                let start = i * self.shape.cols;
                let end = start + self.shape.cols;
                let row_sum: f32 = self.data[start..end].iter().sum();
                row_sum / cols
            })
            .collect();
        Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: 1,
            },
        }
    }

    pub fn tanh(&self) -> Tensor {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn rsqrt_slow(&self) -> Tensor {
        let data = self.data.iter().map(|&x| 1.0 / x.sqrt()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn rsqrt(&self) -> Tensor {
        let data = self
            .data
            .iter()
            .map(|&x| {
                let mut y = f32::from_bits(0x5f3759df - (x.to_bits() >> 1));
                y = y * (1.5 - ((x * 0.5) * y * y)); // One Newton iteration
                y
            })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn rsqrt_simd(&self) -> Tensor {
        let len = self.data.len();
        let mut result_data = Vec::with_capacity(len);

        // Process complete chunks of 4 with direct SIMD
        let complete_chunks = len / 4;
        let remainder = len % 4;

        // Process complete chunks - fastest path
        for i in 0..complete_chunks {
            let start_idx = i * 4;
            unsafe {
                let input: __m128 = _mm_load_ps(self.data.as_ptr().add(start_idx));
                let result: __m128 = _mm_rsqrt_ps(input);
                let output: [f32; 4] = std::mem::transmute(result);
                result_data.extend_from_slice(&output);
            }
        }

        // Handle remainder with padded SIMD
        if remainder > 0 {
            let start_idx = complete_chunks * 4;
            let mut padded = [1.0; 4]; // rsqrt(1.0) = 1.0

            // Copy remaining elements
            for i in 0..remainder {
                padded[i] = self.data[start_idx + i];
            }

            unsafe {
                let input: __m128 = std::mem::transmute(padded);
                let result: __m128 = _mm_rsqrt_ps(input);
                let output: [f32; 4] = std::mem::transmute(result);

                // Only take the elements we need
                result_data.extend_from_slice(&output[..remainder]);
            }
        }

        Tensor {
            data: result_data,
            shape: self.shape.clone(),
        }
    }

    pub fn erf(&self) -> Tensor {
        let data = self.data.iter().map(|x| erf(*x)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn shape(&self) -> Shape {
        Shape {
            cols: self.shape.cols,
            rows: self.shape.rows,
        }
    }

    /// https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
    /// Tricky, it should either return a view or a copy
    /// TODO: for now copy this thing and later on create a view struct to be returned
    pub fn view(&self, rows: Option<usize>, cols: Option<usize>) -> Result<Tensor, TensorError> {
        let total = self.data.len();

        match (rows, cols) {
            (Some(r), Some(c)) => {
                if r * c != total {
                    return Err(TensorError::InvalidDataLength {
                        expected: r * c,
                        found: total,
                    });
                }
                Ok(Tensor {
                    data: self.data.clone(),
                    shape: Shape { rows: r, cols: c },
                })
            }
            (None, Some(c)) => {
                if total % c != 0 {
                    return Err(TensorError::InvalidDataLength {
                        expected: c,
                        found: total,
                    });
                }
                Ok(Tensor {
                    data: self.data.clone(),
                    shape: Shape {
                        rows: total / c,
                        cols: c,
                    },
                })
            }
            (Some(r), None) => {
                if total % r != 0 {
                    return Err(TensorError::InvalidDataLength {
                        expected: r,
                        found: total,
                    });
                }
                Ok(Tensor {
                    data: self.data.clone(),
                    shape: Shape {
                        rows: r,
                        cols: total / r,
                    },
                })
            }
            (None, None) => Err(TensorError::InvalidDataLength {
                expected: total,
                found: 0,
            }),
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        if self.shape.cols != other.shape.rows {
            return Err(TensorError::InvalidMatrixMultiplication {
                left_cols: self.shape.cols,
                right_rows: other.shape.rows,
            });
        }

        let mut result = Tensor::new(self.shape.rows, other.shape.cols);

        for i in 0..self.shape.rows {
            for j in 0..other.shape.cols {
                let mut sum = 0.0;
                for k in 0..self.shape.cols {
                    sum +=
                        self.data[i * self.shape.cols + k] * other.data[k * other.shape.cols + j];
                }
                result.data[i * other.shape.cols + j] = sum;
            }
        }

        Ok(result)
    }

    pub fn transpose(&self) -> Tensor {
        let mut result = Tensor::new(self.shape.cols, self.shape.rows);
        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                result.data[j * self.shape.rows + i] = self.data[i * self.shape.cols + j];
            }
        }
        result
    }

    fn get_broadcasted(&self, row: usize, col: usize) -> f32 {
        let actual_row = if self.shape.rows == 1 { 0 } else { row };
        let actual_col = if self.shape.cols == 1 { 0 } else { col };
        self.data[actual_row * self.shape.cols + actual_col]
    }

    fn broadcast<F: Fn(f32, f32) -> f32 + Clone + Copy>(
        &self,
        other: &Tensor,
        op: F,
    ) -> Result<Tensor, TensorError> {
        let result_shape = self.shape.broadcast(&other.shape)?;
        // let mut data = Vec::with_capacity(result_shape.size());

        let data = (0..result_shape.rows)
            .map(|i| {
                (0..result_shape.cols).map(move |j| {
                    let val1 = self.get_broadcasted(i, j);
                    let val2 = other.get_broadcasted(i, j);
                    op(val1, val2)
                })
            })
            .flatten()
            .collect();

        // for i in 0..result_shape.rows {
        //     for j in 0..result_shape.cols {
        //         let val1 = self.get_broadcasted(i, j);
        //         let val2 = other.get_broadcasted(i, j);
        //         data.push(op(val1,val2));
        //     }
        // }

        Ok(Tensor {
            data,
            shape: result_shape,
        })
    }
}

impl Eq for Tensor {}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl From<Shape> for Tensor {
    fn from(data: Shape) -> Self {
        Self::new(data.rows, data.cols)
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let cols = data.len();
        Self {
            data,
            shape: Shape { rows: 1, cols },
        }
    }
}

impl From<Vec<Vec<f32>>> for Tensor {
    fn from(matrix: Vec<Vec<f32>>) -> Self {
        if matrix.is_empty() {
            return Self::new(0, 0);
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        let data = matrix.into_iter().flatten().collect();

        Self {
            data,
            shape: Shape { rows, cols },
        }
    }
}

impl From<f32> for Tensor {
    fn from(scalar: f32) -> Self {
        Self {
            data: vec![scalar],
            shape: Shape { rows: 1, cols: 1 },
        }
    }
}

impl<const M: usize, const N: usize> From<[[f32; N]; M]> for Tensor {
    /// Create an M×N tensor from a 2D array
    fn from(array: [[f32; N]; M]) -> Self {
        let data = array.into_iter().flatten().collect();
        Self {
            data,
            shape: Shape { rows: M, cols: N },
        }
    }
}

impl Add for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn add(self, other: &Tensor) -> Self::Output {
        if self.shape() == other.shape() {
            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();

            Ok(Tensor {
                data,
                shape: self.shape.clone(),
            })
        } else {
            self.broadcast(other, |a, b| a + b)
        }
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, other: &Tensor) -> Self::Output {
        if self.shape() == other.shape() {
            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect();

            Ok(Tensor {
                data,
                shape: self.shape.clone(),
            })
        } else {
            self.broadcast(other, |a, b| a - b)
        }
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, other: &Tensor) -> Self::Output {
        if self.shape() == other.shape() {
            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect();

            Ok(Tensor {
                data,
                shape: self.shape.clone(),
            })
        } else {
            self.broadcast(&other, |a, b| a * b)
        }
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, other: &Tensor) -> Self::Output {
        //TODO: we assume that we'll never have 0 as value
        if self.shape() == other.shape() {
            let data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a / b)
                .collect();

            Ok(Tensor {
                data,
                shape: self.shape.clone(),
            })
        } else {
            self.broadcast(other, |a, b| a / b)
        }
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    fn add(self, tensor: &Tensor) -> Self::Output {
        let data = tensor.data.iter().map(|x| x + self).collect();
        Tensor {
            data,
            shape: tensor.shape.clone(),
        }
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Self::Output {
        scalar + self
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Self::Output {
        let data = self.data.iter().map(|x| x - scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor) -> Self::Output {
        let data = tensor.data.iter().map(|x| x * self).collect();
        Tensor {
            data,
            shape: tensor.shape.clone(),
        }
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Self::Output {
        scalar * self
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Self::Output {
        let data = self.data.iter().map(|x| x / scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precision = f.precision().unwrap_or(10); // Default to 10 if not set
        write!(f, "Tensor({}x{}) [\n", self.shape.rows, self.shape.cols)?;
        for i in 0..self.shape.rows {
            write!(f, "  [")?;
            for j in 0..self.shape.cols {
                let val = self.data[i * self.shape.cols + j];
                write!(f, "{:.*}", precision, val)?;
                if j < self.shape.cols - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            if i < self.shape.rows - 1 {
                write!(f, ",\n")?;
            }
        }
        write!(f, "\n]")
    }
}

fn reciprocal_sqrt(values: [f32; 4]) -> [f32; 4] {
    unsafe {
        let input: __m128 = std::mem::transmute(values);
        let result: __m128 = _mm_rsqrt_ps(input);
        std::mem::transmute(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor() {
        println!("Tensor: {:?}", Tensor::new(0, 3));

        println!("Tensor: {:?}", Tensor::new(3, 2));
    }

    #[test]
    fn test_view() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(
            a.view(Some(1), None).unwrap(),
            Tensor::from([[1.0, 2.0, 3.0, 4.0]])
        );

        assert_eq!(
            a.view(Some(1), Some(4)).unwrap(),
            Tensor::from([[1.0, 2.0, 3.0, 4.0]])
        );

        assert_eq!(
            a.view(None, Some(1)).unwrap(),
            Tensor::from([[1.0], [2.0], [3.0], [4.0]])
        );

        assert_eq!(
            a.view(Some(4), Some(1)).unwrap(),
            Tensor::from([[1.0], [2.0], [3.0], [4.0]])
        );

        assert!(a.view(Some(5), Some(1)).is_err());

        assert!(a.view(Some(5), Some(5)).is_err());

        assert!(a.view(None, None).is_err());
    }

    #[test]
    fn test_rsqrt() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        // let result = reciprocal_sqrt(a.data()[0..4].try_into().unwrap());

        assert_eq!(
            a.rsqrt(),
            Tensor::from([[0.9983071685, 0.7069300413], [0.5768468380, 0.4991535842]])
        );
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        // No keep dim
        // assert_eq!(a.mean(), Tensor::from(vec![1.5, 3.5]));

        //keep dim
        assert_eq!(a.mean(), Tensor::from([[1.5], [3.5]]));
    }

    #[test]
    fn test_powf() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(a.powf(2.0), Tensor::from([[1.0, 4.0], [9.0, 16.0]]));
    }

    #[test]
    fn test_broadcast_wrong() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[10.0, 20.0, 20.0]]);

        let result = &a + &b;

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_addition() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[10.0, 20.0]]);

        let result = (&a + &b)?;

        let expected = Tensor::from([[11.0, 22.0], [13.0, 24.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_broadcast_subtraction() -> Result<(), TensorError> {
        let a = Tensor::from([[10.0, 20.0], [30.0, 40.0]]);
        let b = Tensor::from([[5.0], [15.0]]);
        let result = (&a - &b)?;

        // b broadcasts to [[5,5], [15,15]], result: [[5,15], [15,25]]
        let expected = Tensor::from([[5.0, 15.0], [15.0, 25.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_broadcast_multiplication() -> Result<(), TensorError> {
        let a = Tensor::from([[2.0, 4.0], [6.0, 8.0]]);
        let b = Tensor::from([[3.0]]);
        let result = (&a * &b)?;

        let expected = Tensor::from([[6.0, 12.0], [18.0, 24.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_broadcast_division() -> Result<(), TensorError> {
        let a = Tensor::from([[10.0, 20.0], [30.0, 40.0]]);
        let b = Tensor::from([[2.0, 4.0]]);

        let result = (&a / &b)?;

        let expected = Tensor::from([[5.0, 5.0], [15.0, 10.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_matmul() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = a.matmul(&b)?;

        let expected = Tensor::from([[7.0, 10.0], [15.0, 22.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_div() -> Result<(), TensorError> {
        let a = Tensor::from([[3.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 1.0], [6.0, 2.0]]);

        let result = (&a / &b)?;

        let expected = Tensor::from([[3.0, 2.0], [0.5, 2.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_mul() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = (&a * &b)?;

        let expected = Tensor::from([[1.0, 4.0], [9.0, 16.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_sub() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = (&a - &b)?;

        let expected = Tensor::from([[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_addition() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = (&a + &b)?;

        let expected = Tensor::from([[2.0, 4.0], [6.0, 8.0]]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_tensor_erf() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let expected = Tensor::from([[0.8427008, 0.9953223], [1.0, 1.0]]);
        assert_eq!(a.erf().data, expected.data);

        Ok(())
    }

    #[test]
    fn test_tensor_tanh() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let expected = Tensor::from([[0.7615942, 0.9640276], [0.9950548, 0.9993293]]);
        assert_eq!(a.tanh().data, expected.data);

        Ok(())
    }

    #[test]
    fn test_triu() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let expected = Tensor::from([[0.0, 2.0], [0.0, 0.0]]);
        assert_eq!(a.triu(1), expected);
    }

    #[test]
    fn test_repeat_interleave() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let result = a.repeat_interleave(2, 0);
        let expected = Tensor::from([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]);
        assert_eq!(result, expected);

        let result = a.repeat_interleave(2, 1);
        let expected = Tensor::from([
            [1.0000000000, 1.0000000000, 2.0000000000, 2.0000000000],
            [3.0000000000, 3.0000000000, 4.0000000000, 4.0000000000],
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_masked_fill() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let mask = Tensor::from([[0.0, 1.0], [1.0, 0.0]]);
        let result = a.masked_fill(&mask, -1.0).unwrap();
        let expected = Tensor::from([[-1.0, 2.0], [3.0, -1.0]]);
        assert_eq!(result, expected);
    }

        #[test]
    fn test_view_transformation() {
        let queries = Tensor::from(
            [[ 0.7383,  0.4590,  0.3379, -0.1719,  0.0024,  0.0132,  0.9023,
          -0.8086, -0.7461, -0.2559,  1.0938, -0.6367,  0.0347,  0.7539,
           0.3105, -0.4219, -0.9453, -0.5312, -0.0771,  0.1904,  0.6211,
          -0.7734,  0.3281,  1.0625, -0.3242, -0.4766,  0.2754, -0.6211,
           0.8516, -0.3418, -1.0078,  0.3438],
         [ 0.2852,  0.3242, -0.0037, -0.0498,  0.8945, -0.0322,  0.8438,
           0.1328, -0.7383,  0.6289,  0.8906,  0.8555,  0.5586,  0.4141,
           0.9023, -0.5781, -0.6289, -0.5547,  0.6992,  0.4473,  1.0000,
          -0.4785,  0.0991, -0.0649,  0.7773,  0.2021, -0.2559, -0.4922,
           0.7617,  0.0240, -0.8398,  0.3203],
         [-0.7422, -0.4316, -0.2637,  0.1719, -0.2812,  0.0791, -1.0547,
           0.6055,  0.8555,  0.0452, -1.1953,  0.2041, -0.1562, -0.8359,
          -0.4766,  0.5117,  0.9414,  0.6680, -0.1699, -0.2852, -0.7852,
           0.7930, -0.3770, -0.8047,  0.0061,  0.3047, -0.1533,  0.6289,
          -0.9336,  0.3418,  1.0469, -0.4727]]
        );

        let queries = queries.view(1, 3, 4, 8).unwrap();

        let expected = Tensor::from(
            [[[ 0.7383,  0.4590,  0.3379, -0.1719,  0.0024,  0.0132,  0.9023,
           -0.8086],
          [-0.7461, -0.2559,  1.0938, -0.6367,  0.0347,  0.7539,  0.3105,
           -0.4219],
          [-0.9453, -0.5312, -0.0771,  0.1904,  0.6211, -0.7734,  0.3281,
            1.0625],
          [-0.3242, -0.4766,  0.2754, -0.6211,  0.8516, -0.3418, -1.0078,
            0.3438]],

         [[ 0.2852,  0.3242, -0.0037, -0.0498,  0.8945, -0.0322,  0.8438,
            0.1328],
          [-0.7383,  0.6289,  0.8906,  0.8555,  0.5586,  0.4141,  0.9023,
           -0.5781],
          [-0.6289, -0.5547,  0.6992,  0.4473,  1.0000, -0.4785,  0.0991,
           -0.0649],
          [ 0.7773,  0.2021, -0.2559, -0.4922,  0.7617,  0.0240, -0.8398,
            0.3203]],

         [[-0.7422, -0.4316, -0.2637,  0.1719, -0.2812,  0.0791, -1.0547,
            0.6055],
          [ 0.8555,  0.0452, -1.1953,  0.2041, -0.1562, -0.8359, -0.4766,
            0.5117],
          [ 0.9414,  0.6680, -0.1699, -0.2852, -0.7852,  0.7930, -0.3770,
           -0.8047],
          [ 0.0061,  0.3047, -0.1533,  0.6289, -0.9336,  0.3418,  1.0469,
           -0.4727]]]
        );

        let queries = queries.transpose(1, 2);


        let transpose_expected = Tensor::from(
            [[[ 0.7383,  0.4590,  0.3379, -0.1719,  0.0024,  0.0132,  0.9023,
           -0.8086],
          [ 0.2852,  0.3242, -0.0037, -0.0498,  0.8945, -0.0322,  0.8438,
            0.1328],
          [-0.7422, -0.4316, -0.2637,  0.1719, -0.2812,  0.0791, -1.0547,
            0.6055]],

         [[-0.7461, -0.2559,  1.0938, -0.6367,  0.0347,  0.7539,  0.3105,
           -0.4219],
          [-0.7383,  0.6289,  0.8906,  0.8555,  0.5586,  0.4141,  0.9023,
           -0.5781],
          [ 0.8555,  0.0452, -1.1953,  0.2041, -0.1562, -0.8359, -0.4766,
            0.5117]],

         [[-0.9453, -0.5312, -0.0771,  0.1904,  0.6211, -0.7734,  0.3281,
            1.0625],
          [-0.6289, -0.5547,  0.6992,  0.4473,  1.0000, -0.4785,  0.0991,
           -0.0649],
          [ 0.9414,  0.6680, -0.1699, -0.2852, -0.7852,  0.7930, -0.3770,
           -0.8047]],

         [[-0.3242, -0.4766,  0.2754, -0.6211,  0.8516, -0.3418, -1.0078,
            0.3438],
          [ 0.7773,  0.2021, -0.2559, -0.4922,  0.7617,  0.0240, -0.8398,
            0.3203],
          [ 0.0061,  0.3047, -0.1533,  0.6289, -0.9336,  0.3418,  1.0469,
           -0.4727]]]
        );
    }
}
