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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Broadcast two shapes (like PyTorch)
    pub fn broadcast(&self, other: &Shape) -> Result<Shape, TensorError> {
        let mut result = Vec::new();
        let a = &self.0;
        let b = &other.0;
        let ndim = a.len().max(b.len());
        for i in 0..ndim {
            let adim = *a.get(a.len().wrapping_sub(i + 1)).unwrap_or(&1);
            let bdim = *b.get(b.len().wrapping_sub(i + 1)).unwrap_or(&1);
            if adim == bdim || adim == 1 || bdim == 1 {
                result.push(adim.max(bdim));
            } else {
                return Err(TensorError::DimensionMismatch {
                    expected: (adim, bdim),
                    found: (adim, bdim),
                });
            }
        }
        result.reverse();
        Ok(Shape::new(result))
    }
}

pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
     pub fn exp(&self) -> Result<Tensor, TensorError> {
        let data = self.data.iter().map(|x| x.exp()).collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

     pub fn softmax(&self, axis: isize) -> Result<Tensor, TensorError> {
        let ndim = self.shape.0.len();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };
        if axis >= ndim {
            return Err(TensorError::DimensionMismatch {
                expected: (ndim, axis),
                found: (ndim, axis),
            });
        }
        let max = self.max_dim(axis, true)?;
        let exp = (self - &max)?.exp()?;
        let sum = exp.sum_dim(axis, true)?;
        &exp / &sum
    }

    //TODO: test
     pub fn max_dim(&self, axis: usize, keepdim: bool) -> Result<Tensor, TensorError> {
        let shape = &self.shape.0;
        let mut out_shape = shape.clone();
        if keepdim {
            out_shape[axis] = 1;
        } else {
            out_shape.remove(axis);
        }
        let strides = compute_strides(shape);
        let out_strides = compute_strides(&out_shape);

        let mut out_data = vec![f32::NEG_INFINITY; out_shape.iter().product()];
        for idx in 0..self.data.len() {
            let indices = unravel_index(idx, &strides, shape);
            let mut out_indices = indices.clone();
            if keepdim {
                out_indices[axis] = 0;
            } else {
                out_indices.remove(axis);
            }
            let out_idx = ravel_index(&out_indices, &out_strides, &out_shape);
            out_data[out_idx] = out_data[out_idx].max(self.data[idx]);
        }
        Ok(Tensor {
            data: out_data,
            shape: Shape::new(out_shape),
        })
    }

    //TODO: test
    pub fn sum_dim(&self, axis: usize, keepdim: bool) -> Result<Tensor, TensorError> {
        let shape = &self.shape.0;
        let mut out_shape = shape.clone();
        if keepdim {
            out_shape[axis] = 1;
        } else {
            out_shape.remove(axis);
        }
        let strides = compute_strides(shape);
        let out_strides = compute_strides(&out_shape);

        let mut out_data = vec![0.0; out_shape.iter().product()];
        for idx in 0..self.data.len() {
            let indices = unravel_index(idx, &strides, shape);
            let mut out_indices = indices.clone();
            if keepdim {
                out_indices[axis] = 0;
            } else {
                out_indices.remove(axis);
            }
            let out_idx = ravel_index(&out_indices, &out_strides, &out_shape);
            out_data[out_idx] += self.data[idx];
        }
        Ok(Tensor {
            data: out_data,
            shape: Shape::new(out_shape),
        })
    }

    pub fn new(dims: &[usize]) -> Self {
        let numel = dims.iter().product();
        Self {
            data: vec![Default::default(); numel],
            shape: Shape::new(dims.to_vec()),
        }
    }

    pub fn ones(dims: &[usize]) -> Self {
        let numel = dims.iter().product();
        Self {
            data: vec![1.0; numel],
            shape: Shape::new(dims.to_vec()),
        }
    }

    pub fn triu(&self, k: isize) -> Tensor {
        let shape = &self.shape.0;
        assert!(shape.len() == 2, "triu only supports 2D tensors");
        let rows = shape[0];
        let cols = shape[1];
        let mut data = self.data.clone();
        for i in 0..rows {
            for j in 0..cols {
                if (j as isize) - (i as isize) < k {
                    data[i * cols + j] = 0.0;
                }
            }
        }
        Tensor {
            data,
            shape: Shape::new(vec![rows, cols]),
        }
    }

    pub fn repeat_interleave(&self, repeats: usize, dim: usize) -> Tensor {
        let shape = &self.shape.0;
        let mut new_shape = shape.clone();
        new_shape[dim] *= repeats;
        let mut data = Vec::with_capacity(self.data.len() * repeats);
        let strides = compute_strides(shape);
        let new_strides = compute_strides(&new_shape);
        for idx in 0..self.data.len() {
            let indices = unravel_index(idx, &strides, shape);
            for r in 0..repeats {
                let mut new_indices = indices.clone();
                new_indices[dim] = indices[dim] * repeats + r;
                let new_idx = ravel_index(&new_indices, &new_strides, &new_shape);
                if new_idx >= data.len() {
                    data.resize(new_idx + 1, 0.0);
                }
                data[new_idx] = self.data[idx];
            }
        }
        Tensor {
            data,
            shape: Shape::new(new_shape),
        }
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Tensor, TensorError> {
        if self.shape != mask.shape {
            return Err(TensorError::DimensionMismatch {
                expected: (self.shape.num_elements(), 1),
                found: (mask.shape.num_elements(), 1),
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

    pub fn rand(dims: &[usize], seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let numel = dims.iter().product();
        let mut data = vec![Default::default(); numel];
        data.fill_with(|| rng.random());
        Self {
            data,
            shape: Shape::new(dims.to_vec()),
        }
    }

    pub fn powf(&self, exp: f32) -> Tensor {
        let data = self.data.iter().map(|x| x.powf(exp)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn mean(&self, axis: Option<usize>) -> Tensor {
        let shape = &self.shape.0;
        if shape.len() == 2 && axis.is_none() {
            let cols = shape[1] as f32;
            let data = (0..shape[0])
                .map(|i| {
                    let start = i * shape[1];
                    let end = start + shape[1];
                    let row_sum: f32 = self.data[start..end].iter().sum();
                    row_sum / cols
                })
                .collect();
            Tensor {
                data,
                shape: Shape::new(vec![shape[0], 1]),
            }
        } else {
            unimplemented!("mean for N-dim tensors and axis reduction");
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
                y = y * (1.5 - ((x * 0.5) * y * y));
                y
            })
            .collect();
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

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn view(&self, dims: &[usize]) -> Result<Tensor, TensorError> {
        let total = dims.iter().product();
        if total != self.data.len() {
            return Err(TensorError::InvalidDataLength {
                expected: total,
                found: self.data.len(),
            });
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(dims.to_vec()),
        })
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let mut new_shape = self.shape.0.clone();
        new_shape.swap(dim0, dim1);
        let old_strides = compute_strides(&self.shape.0);
        let new_strides = compute_strides(&new_shape);
        let mut new_data = vec![0.0; self.data.len()];
        for idx in 0..self.data.len() {
            let mut old_idx = idx;
            let mut indices = vec![0; self.shape.0.len()];
            for (i, stride) in old_strides.iter().enumerate() {
                indices[i] = old_idx / stride;
                old_idx %= stride;
            }
            indices.swap(dim0, dim1);
            let mut new_idx = 0;
            for (i, stride) in new_strides.iter().enumerate() {
                new_idx += indices[i] * stride;
            }
            new_data[new_idx] = self.data[idx];
        }
        Tensor {
            data: new_data,
            shape: Shape::new(new_shape),
        }
    }

    fn get_broadcasted(&self, indices: &[usize]) -> f32 {
        let mut actual_indices = indices.to_vec();
        for (i, &dim) in self.shape.0.iter().enumerate() {
            if dim == 1 {
                actual_indices[i] = 0;
            }
        }
        let idx = ravel_index(
            &actual_indices,
            &compute_strides(&self.shape.0),
            &self.shape.0,
        );
        self.data[idx]
    }

    fn broadcast<F: Fn(f32, f32) -> f32 + Clone + Copy>(
        &self,
        other: &Tensor,
        op: F,
    ) -> Result<Tensor, TensorError> {
        let result_shape = self.shape.broadcast(&other.shape)?;
        let data = (0..result_shape.num_elements())
            .map(|idx| {
                let indices =
                    unravel_index(idx, &compute_strides(&result_shape.0), &result_shape.0);
                let val1 = self.get_broadcasted(&indices);
                let val2 = other.get_broadcasted(&indices);
                op(val1, val2)
            })
            .collect();
        Ok(Tensor {
            data,
            shape: result_shape,
        })
    }

    /// Matrix multiplication (supports 1D, 2D, and batched N-D tensors)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let a_shape = &self.shape.0;
        let b_shape = &other.shape.0;
        match (a_shape.len(), b_shape.len()) {
            // 1D x 1D: dot product
            (1, 1) => {
                if a_shape[0] != b_shape[0] {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: a_shape[0],
                        right_rows: b_shape[0],
                    });
                }
                let dot: f32 = self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum();
                Ok(Tensor {
                    data: vec![dot],
                    shape: Shape::new(vec![1]),
                })
            }
            // 2D x 2D: matrix multiplication
            (2, 2) => {
                let (m, k1) = (a_shape[0], a_shape[1]);
                let (k2, n) = (b_shape[0], b_shape[1]);
                if k1 != k2 {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: k1,
                        right_rows: k2,
                    });
                }
                let mut data = vec![0.0; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..k1 {
                            sum += self.data[i * k1 + k] * other.data[k * n + j];
                        }
                        data[i * n + j] = sum;
                    }
                }
                Ok(Tensor {
                    data,
                    shape: Shape::new(vec![m, n]),
                })
            }
            // Batched matmul: N-D tensors
            (a_ndim, b_ndim) if a_ndim >= 2 && b_ndim >= 2 => {
                // Broadcast batch dimensions
                let a_batch = &a_shape[..a_ndim - 2];
                let b_batch = &b_shape[..b_ndim - 2];
                let batch_shape = Shape::new(a_batch.to_vec())
                    .broadcast(&Shape::new(b_batch.to_vec()))?
                    .0;
                let (m, k1) = (a_shape[a_ndim - 2], a_shape[a_ndim - 1]);
                let (k2, n) = (b_shape[b_ndim - 2], b_shape[b_ndim - 1]);
                if k1 != k2 {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: k1,
                        right_rows: k2,
                    });
                }
                let batch_size = batch_shape.iter().product::<usize>();
                let mut data = vec![0.0; batch_size * m * n];
                let a_strides = compute_strides(a_shape);
                let b_strides = compute_strides(b_shape);
                let out_strides = compute_strides(&[batch_shape.clone(), vec![m, n]].concat());
                for batch_idx in 0..batch_size {
                    let batch_indices =
                        unravel_index(batch_idx, &compute_strides(&batch_shape), &batch_shape);
                    // Find corresponding indices in a and b (broadcasted)
                    let mut a_batch_indices = vec![0; a_batch.len()];
                    let mut b_batch_indices = vec![0; b_batch.len()];
                    for i in 0..batch_shape.len() {
                        a_batch_indices[i] = if i < a_batch.len() && a_batch[i] == 1 {
                            0
                        } else {
                            batch_indices[i]
                        };
                        b_batch_indices[i] = if i < b_batch.len() && b_batch[i] == 1 {
                            0
                        } else {
                            batch_indices[i]
                        };
                    }
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for k in 0..k1 {
                                // Build full index for a: [a_batch..., i, k]
                                let mut a_idx = a_batch_indices.clone();
                                a_idx.push(i);
                                a_idx.push(k);
                                let a_flat = ravel_index(&a_idx, &a_strides, a_shape);
                                // Build full index for b: [b_batch..., k, j]
                                let mut b_idx = b_batch_indices.clone();
                                b_idx.push(k);
                                b_idx.push(j);
                                let b_flat = ravel_index(&b_idx, &b_strides, b_shape);
                                sum += self.data[a_flat] * other.data[b_flat];
                            }
                            // Output index: [batch..., i, j]
                            let mut out_idx = batch_indices.clone();
                            out_idx.push(i);
                            out_idx.push(j);
                            let out_flat = ravel_index(
                                &out_idx,
                                &out_strides,
                                &[batch_shape.clone(), vec![m, n]].concat(),
                            );
                            data[out_flat] = sum;
                        }
                    }
                }
                Ok(Tensor {
                    data,
                    shape: Shape::new([batch_shape, vec![m, n]].concat()),
                })
            }
            _ => Err(TensorError::InvalidMatrixMultiplication {
                left_cols: *a_shape.last().unwrap_or(&0),
                right_rows: *b_shape.first().unwrap_or(&0),
            }),
        }
    }
}

// Helper functions for N-dim indexing
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn unravel_index(mut idx: usize, strides: &[usize], shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for i in 0..shape.len() {
        indices[i] = idx / strides[i];
        idx %= strides[i];
    }
    indices
}

fn ravel_index(indices: &[usize], strides: &[usize], _shape: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

impl Eq for Tensor {}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

// Helper trait for recursive conversion from nested arrays/slices
pub trait TensorFromNested {
    fn flatten_and_shape(&self) -> (Vec<f32>, Vec<usize>);
}

impl TensorFromNested for f32 {
    fn flatten_and_shape(&self) -> (Vec<f32>, Vec<usize>) {
        (vec![*self], vec![])
    }
}

impl<T: TensorFromNested> TensorFromNested for &[T] {
    fn flatten_and_shape(&self) -> (Vec<f32>, Vec<usize>) {
        if self.is_empty() {
            return (vec![], vec![0]);
        }
        let mut shape = vec![self.len()];
        let (first_flat, first_shape) = self[0].flatten_and_shape();
        let mut flat = first_flat;
        for item in &self[1..] {
            let (item_flat, item_shape) = item.flatten_and_shape();
            assert_eq!(
                item_shape, first_shape,
                "All subarrays must have the same shape"
            );
            flat.extend(item_flat);
        }
        shape.extend(first_shape);
        (flat, shape)
    }
}

impl<T: TensorFromNested, const N: usize> TensorFromNested for [T; N] {
    fn flatten_and_shape(&self) -> (Vec<f32>, Vec<usize>) {
        self.as_slice().flatten_and_shape()
    }
}

impl<T: TensorFromNested> TensorFromNested for Vec<T> {
    fn flatten_and_shape(&self) -> (Vec<f32>, Vec<usize>) {
        self.as_slice().flatten_and_shape()
    }
}

impl<T: TensorFromNested> From<T> for Tensor {
    fn from(nested: T) -> Self {
        let (data, shape) = nested.flatten_and_shape();
        let shape = if shape.is_empty() {
            vec![1, data.len()] // Always create [1, len] for flat vectors
        } else {
            shape
        };
        Tensor {
            data,
            shape: Shape::new(shape),
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
            self.broadcast(other, |a, b| a * b)
        }
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor, TensorError>;

    fn div(self, other: &Tensor) -> Self::Output {
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
        let precision = f.precision().unwrap_or(10);
        write!(f, "Tensor({:?}) [\n", self.shape.dims())?;
        for i in 0..self.shape.num_elements() {
            if i % self.shape.dims().last().unwrap() == 0 {
                write!(f, "  [")?;
            }
            let val = self.data[i];
            write!(f, "{:.*}", precision, val)?;
            if (i + 1) % self.shape.dims().last().unwrap() == 0 {
                write!(f, "]")?;
                if i < self.shape.num_elements() - 1 {
                    write!(f, ",\n")?;
                }
            } else {
                write!(f, ", ")?;
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
        println!("Tensor: {:?}", Tensor::new(&[0, 3]));

        println!("Tensor: {:?}", Tensor::new(&[3, 2]));
    }

    #[test]
    fn test_view() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(
            a.view(&[1, 4]).unwrap(),
            Tensor::from([[1.0, 2.0, 3.0, 4.0]])
        );

        assert_eq!(
            a.view(&[4, 1]).unwrap(),
            Tensor::from([[1.0], [2.0], [3.0], [4.0]])
        );

        assert!(a.view(&[5, 1]).is_err());

        assert!(a.view(&[5, 5]).is_err());
    }

    #[test]
    fn test_rsqrt() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(
            a.rsqrt(),
            Tensor::from([[0.9983071685, 0.7069300413], [0.5768468380, 0.4991535842]])
        );
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(a.mean(None), Tensor::from([[1.5], [3.5]]));
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
        let queries = Tensor::from([[
            [
                0.7383, 0.4590, 0.3379, -0.1719, 0.0024, 0.0132, 0.9023, -0.8086, -0.7461, -0.2559,
                1.0938, -0.6367, 0.0347, 0.7539, 0.3105, -0.4219, -0.9453, -0.5312, -0.0771,
                0.1904, 0.6211, -0.7734, 0.3281, 1.0625, -0.3242, -0.4766, 0.2754, -0.6211, 0.8516,
                -0.3418, -1.0078, 0.3438,
            ],
            [
                0.2852, 0.3242, -0.0037, -0.0498, 0.8945, -0.0322, 0.8438, 0.1328, -0.7383, 0.6289,
                0.8906, 0.8555, 0.5586, 0.4141, 0.9023, -0.5781, -0.6289, -0.5547, 0.6992, 0.4473,
                1.0000, -0.4785, 0.0991, -0.0649, 0.7773, 0.2021, -0.2559, -0.4922, 0.7617, 0.0240,
                -0.8398, 0.3203,
            ],
            [
                -0.7422, -0.4316, -0.2637, 0.1719, -0.2812, 0.0791, -1.0547, 0.6055, 0.8555,
                0.0452, -1.1953, 0.2041, -0.1562, -0.8359, -0.4766, 0.5117, 0.9414, 0.6680,
                -0.1699, -0.2852, -0.7852, 0.7930, -0.3770, -0.8047, 0.0061, 0.3047, -0.1533,
                0.6289, -0.9336, 0.3418, 1.0469, -0.4727,
            ],
        ]]);

        assert_eq!(queries.shape(), &Shape::new(vec![1, 3, 32]));

        let queries = queries.view(&[1, 3, 4, 8]).unwrap();

        let expected = Tensor::from([[
            [
                [
                    0.7383, 0.4590, 0.3379, -0.1719, 0.0024, 0.0132, 0.9023, -0.8086,
                ],
                [
                    -0.7461, -0.2559, 1.0938, -0.6367, 0.0347, 0.7539, 0.3105, -0.4219,
                ],
                [
                    -0.9453, -0.5312, -0.0771, 0.1904, 0.6211, -0.7734, 0.3281, 1.0625,
                ],
                [
                    -0.3242, -0.4766, 0.2754, -0.6211, 0.8516, -0.3418, -1.0078, 0.3438,
                ],
            ],
            [
                [
                    0.2852, 0.3242, -0.0037, -0.0498, 0.8945, -0.0322, 0.8438, 0.1328,
                ],
                [
                    -0.7383, 0.6289, 0.8906, 0.8555, 0.5586, 0.4141, 0.9023, -0.5781,
                ],
                [
                    -0.6289, -0.5547, 0.6992, 0.4473, 1.0000, -0.4785, 0.0991, -0.0649,
                ],
                [
                    0.7773, 0.2021, -0.2559, -0.4922, 0.7617, 0.0240, -0.8398, 0.3203,
                ],
            ],
            [
                [
                    -0.7422, -0.4316, -0.2637, 0.1719, -0.2812, 0.0791, -1.0547, 0.6055,
                ],
                [
                    0.8555, 0.0452, -1.1953, 0.2041, -0.1562, -0.8359, -0.4766, 0.5117,
                ],
                [
                    0.9414, 0.6680, -0.1699, -0.2852, -0.7852, 0.7930, -0.3770, -0.8047,
                ],
                [
                    0.0061, 0.3047, -0.1533, 0.6289, -0.9336, 0.3418, 1.0469, -0.4727,
                ],
            ],
        ]]);

        assert_eq!(queries, expected);

        let queries = queries.transpose(1, 2);

        let transpose_expected = Tensor::from([[
            [
                [
                    0.7383, 0.4590, 0.3379, -0.1719, 0.0024, 0.0132, 0.9023, -0.8086,
                ],
                [
                    0.2852, 0.3242, -0.0037, -0.0498, 0.8945, -0.0322, 0.8438, 0.1328,
                ],
                [
                    -0.7422, -0.4316, -0.2637, 0.1719, -0.2812, 0.0791, -1.0547, 0.6055,
                ],
            ],
            [
                [
                    -0.7461, -0.2559, 1.0938, -0.6367, 0.0347, 0.7539, 0.3105, -0.4219,
                ],
                [
                    -0.7383, 0.6289, 0.8906, 0.8555, 0.5586, 0.4141, 0.9023, -0.5781,
                ],
                [
                    0.8555, 0.0452, -1.1953, 0.2041, -0.1562, -0.8359, -0.4766, 0.5117,
                ],
            ],
            [
                [
                    -0.9453, -0.5312, -0.0771, 0.1904, 0.6211, -0.7734, 0.3281, 1.0625,
                ],
                [
                    -0.6289, -0.5547, 0.6992, 0.4473, 1.0000, -0.4785, 0.0991, -0.0649,
                ],
                [
                    0.9414, 0.6680, -0.1699, -0.2852, -0.7852, 0.7930, -0.3770, -0.8047,
                ],
            ],
            [
                [
                    -0.3242, -0.4766, 0.2754, -0.6211, 0.8516, -0.3418, -1.0078, 0.3438,
                ],
                [
                    0.7773, 0.2021, -0.2559, -0.4922, 0.7617, 0.0240, -0.8398, 0.3203,
                ],
                [
                    0.0061, 0.3047, -0.1533, 0.6289, -0.9336, 0.3418, 1.0469, -0.4727,
                ],
            ],
        ]]);

        assert_eq!(queries, transpose_expected);
    }
}
