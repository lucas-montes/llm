use std::{
    fmt,
    ops::{Add, Div, Mul, Sub},
};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
    fn broadcast(&self, other: &Shape) -> Result<Shape, TensorError> {
        if self.rows == other.rows && self.cols == other.cols {
            Ok(self.clone())
        } else if self.rows == 1 && self.cols == other.cols {
            Ok(Shape {
                rows: other.rows,
                cols: other.cols,
            })
        } else if self.cols == 1 && self.rows == other.rows {
            Ok(Shape {
                rows: other.rows,
                cols: other.cols,
            })
        } else {
            Err(TensorError::DimensionMismatch {
                expected: (self.rows, self.cols),
                found: (other.rows, other.cols),
            })
        }
    }
}

pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: Vec::with_capacity(rows * cols),
            shape: Shape { rows, cols },
        }
    }

    pub fn rand(rows: usize, cols: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let mut data = Vec::with_capacity(rows * cols);
        data.fill_with(|| rng.random());

        Self {
            data,
            shape: Shape { rows, cols },
        }
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<f32, TensorError> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return Err(TensorError::IndexOutOfBounds {
                row,
                col,
                shape:self.shape.clone(),
            });
        }
        Ok(self.data[row * self.shape.cols + col])
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f32) -> Result<(), TensorError> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return Err(TensorError::IndexOutOfBounds {
                row,
                col,
                shape: self.shape.clone(),
            });
        }
        self.data[row * self.shape.cols + col] = value;
        Ok(())
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

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.sum() / (self.data.len() as f32)
    }

    /// Element-wise power
    pub fn pow(&self, exponent: f32) -> Tensor {
        let data: Vec<_> = self.data.iter().map(|x| x.powf(exponent)).collect();
        Tensor::from(data)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Tensor {
        let data: Vec<_> = self.data.iter().map(|x| x.sqrt()).collect();
        Tensor::from(data)
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
        let data = if self.shape() == other.shape() {
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            todo!()
        };

        Ok(Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
        })
    }
}

// Element-wise subtraction
impl Sub for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn sub(self, other: &Tensor) -> Self::Output {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
        })
    }
}

// Element-wise multiplication (Hadamard product)
impl Mul for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn mul(self, other: &Tensor) -> Self::Output {
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
        })
    }
}

// Element-wise division
impl Div for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, other: &Tensor) -> Self::Output {
        let data: Result<Vec<f32>, TensorError> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                if b.abs() < f32::EPSILON {
                    Err(TensorError::DivisionByZero)
                } else {
                    Ok(a / b)
                }
            })
            .collect();

        Ok(Tensor {
            data: data?,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
        })
    }
}

// Scalar operations
impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, scalar: f32) -> Self::Output {
        let data = self.data.iter().map(|x| x + scalar).collect();
        Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
        }
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, scalar: f32) -> Self::Output {
        let data: Vec<f32> = self.data.iter().map(|x| x - scalar).collect();
        Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
        }
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Self::Output {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
        }
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, scalar: f32) -> Self::Output {
        let data: Vec<f32> = self.data.iter().map(|x| x / scalar).collect();
        Tensor {
            data,
            shape: Shape {
                rows: self.shape.rows,
                cols: self.shape.cols,
            },
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
    fn test_tensor_mean() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(a.mean(), 2.5);
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
}
