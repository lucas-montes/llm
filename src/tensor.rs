use std::{
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

    pub fn tanh(&self) -> Tensor {
        let data = self.data.iter().map(|x| x.tanh()).collect();
        Tensor {
            data,
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

    /// Transpose the tensor
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

impl From<Shape> for Tensor {
    fn from(data: Shape) -> Self {
        Self::new(data.rows, data.cols)
    }
}

// From trait implementations
impl From<Vec<f32>> for Tensor {
    /// Create a row vector (1×n) from Vec<f32>
    fn from(data: Vec<f32>) -> Self {
        let cols = data.len();
        Self {
            data,
            shape: Shape { rows: 1, cols },
        }
    }
}

impl From<Vec<Vec<f32>>> for Tensor {
    /// Create a tensor from 2D vector
    fn from(matrix: Vec<Vec<f32>>) -> Self {
        if matrix.is_empty() {
            return Self::new(0, 0);
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        // Flatten the 2D vector
        let data = matrix.into_iter().flatten().collect();

        Self {
            data,
            shape: Shape { rows, cols },
        }
    }
}

impl From<f32> for Tensor {
    /// Create a 1×1 tensor from a scalar
    fn from(scalar: f32) -> Self {
        Self {
            data: vec![scalar],
            shape: Shape { rows: 1, cols: 1 },
        }
    }
}

// Generic implementation for any 2D array (requires const generics)
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

// Element-wise addition
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

// Element-wise subtraction
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

// Element-wise multiplication (Hadamard product)
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
            self.broadcast(other, |a, b| a * b)
        }
    }
}

// Element-wise division
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
        let data = self.data.iter().map(|x| x + scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
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
        let data = self.data.iter().map(|x| x * scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
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

// Debug implementation
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({}x{}) [\n", self.shape.rows, self.shape.cols)?;
        for i in 0..self.shape.rows {
            write!(f, "  [")?;
            for j in 0..self.shape.cols {
                let val = self.data[i * self.shape.cols + j];
                write!(f, "{:8.4}", val)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor() {
        println!("Tensor: {:?}", Tensor::new(0, 3));

        println!("Tensor: {:?}", Tensor::new(3, 2));
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
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_broadcast_subtraction() -> Result<(), TensorError> {
        let a = Tensor::from([[10.0, 20.0], [30.0, 40.0]]);
        let b = Tensor::from([[5.0], [15.0]]);
        let result = (&a - &b)?;

        // b broadcasts to [[5,5], [15,15]], result: [[5,15], [15,25]]
        let expected = Tensor::from([[5.0, 15.0], [15.0, 25.0]]);
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_broadcast_multiplication() -> Result<(), TensorError> {
        let a = Tensor::from([[2.0, 4.0], [6.0, 8.0]]);
        let b = Tensor::from([[3.0]]);
        let result = (&a * &b)?;

        let expected = Tensor::from([[6.0, 12.0], [18.0, 24.0]]);
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_broadcast_division() -> Result<(), TensorError> {
        let a = Tensor::from([[10.0, 20.0], [30.0, 40.0]]);
        let b = Tensor::from([[2.0, 4.0]]);

        let result = (&a / &b)?;

        let expected = Tensor::from([[5.0, 5.0], [15.0, 10.0]]);
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_tensor_matmul() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = &a.matmul(&b)?;

        let expected = Tensor::from([[7.0, 10.0], [15.0, 22.0]]);
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_tensor_div() -> Result<(), TensorError> {
        let a = Tensor::from([[3.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 1.0], [6.0, 2.0]]);

        let result = (&a / &b)?;

        let expected = Tensor::from([[3.0, 2.0], [0.5, 2.0]]);
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_tensor_mul() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = (&a * &b)?;

        let expected = Tensor::from([[1.0, 4.0], [9.0, 16.0]]);
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_tensor_sub() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = (&a - &b)?;

        let expected = Tensor::from([[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(result.data, expected.data);

        Ok(())
    }

    #[test]
    fn test_tensor_addition() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = (&a + &b)?;

        let expected = Tensor::from([[2.0, 4.0], [6.0, 8.0]]);
        assert_eq!(result.data, expected.data);

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
}
