use std::{
    // arch::x86_64::{__m128, _mm_load_ps, _mm_rsqrt_ps},
    fmt,
    ops::{Add, Div, Mul, Sub},
};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::{iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator
}, slice::ParallelSliceMut};

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

    // Helper function to iterate through all valid indices
    fn extract_slice(
        tensor: &Tensor,
        ranges: &[std::ops::Range<usize>],
        current_indices: &mut Vec<usize>,
        dim: usize,
        new_data: &mut Vec<f32>,
    ) {
        if dim == ranges.len() {
            // Calculate flat index and extract value
            let strides = compute_strides(tensor.shape().dims());
            let idx: usize = current_indices
                .iter()
                .zip(strides.iter())
                .map(|(i, s)| i * s)
                .sum();
            new_data.push(tensor.data()[idx]);
            return;
        }

        for i in ranges[dim].clone() {
            current_indices.push(i);
            Self::extract_slice(tensor, ranges, current_indices, dim + 1, new_data);
            current_indices.pop();
        }
    }

    /// Slice a tensor along multiple dimensions
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Tensor, TensorError> {
        //TODO: test and improve
        let dims = self.shape().dims();
        assert_eq!(
            ranges.len(),
            dims.len(),
            "Number of ranges must match tensor dimensions"
        );

        // Calculate new shape
        let new_shape: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        let mut new_data = Vec::new();

        let mut indices = Vec::new();
        Self::extract_slice(self, ranges, &mut indices, 0, &mut new_data);

        Ok(Tensor {
            data: new_data,
            shape: Shape::new(new_shape),
        })
    }

    /// Add a dimension of size 1 at the specified position
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor, TensorError> {
        //TODO: test and improve
        let mut new_shape = self.shape().dims().to_vec();
        new_shape.insert(dim, 1);

        Ok(Tensor {
            data: self.data.clone(),
            shape: Shape::new(new_shape),
        })
    }

    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Tensor], dim: isize) -> Result<Tensor, TensorError> {
        //TODO: test and improve
        if tensors.is_empty() {
            return Err(TensorError::InvalidDataLength {
                expected: 1,
                found: 0,
            });
        }

        let first_shape = tensors[0].shape().dims();
        let ndim = first_shape.len();
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        // Verify all tensors have compatible shapes
        for tensor in tensors.iter().skip(1) {
            let shape = tensor.shape().dims();
            if shape.len() != ndim {
                return Err(TensorError::DimensionMismatch {
                    expected: (ndim, ndim),
                    found: (shape.len(), ndim),
                });
            }
            for (i, (&d1, &d2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    return Err(TensorError::DimensionMismatch {
                        expected: (d1, d1),
                        found: (d2, d1),
                    });
                }
            }
        }

        // Calculate output shape
        let mut out_shape = first_shape.to_vec();
        out_shape[dim] = tensors.iter().map(|t| t.shape().dims()[dim]).sum();

        // Properly concatenate data along the specified dimension
        let mut out_data = Vec::with_capacity(out_shape.iter().product());

        // For the last dimension (most common case), we can optimize
        if dim == ndim - 1 {
            let outer_size = first_shape[..ndim - 1].iter().product::<usize>();
            for outer_idx in 0..outer_size {
                for tensor in tensors {
                    let start = outer_idx * tensor.shape().dims()[dim];
                    let end = start + tensor.shape().dims()[dim];
                    out_data.extend_from_slice(&tensor.data()[start..end]);
                }
            }
        } else {
            // General case for other dimensions
            let out_strides = compute_strides(&out_shape);

            for out_idx in 0..out_shape.iter().product::<usize>() {
                let out_indices = unravel_index(out_idx, &out_strides, &out_shape);

                // Find which tensor this element comes from
                let mut cumulative_size = 0;
                let mut source_tensor_idx = 0;
                let mut source_dim_idx = out_indices[dim];

                for (i, tensor) in tensors.iter().enumerate() {
                    let tensor_dim_size = tensor.shape().dims()[dim];
                    if source_dim_idx < cumulative_size + tensor_dim_size {
                        source_tensor_idx = i;
                        source_dim_idx -= cumulative_size;
                        break;
                    }
                    cumulative_size += tensor_dim_size;
                }

                // Build source indices
                let mut source_indices = out_indices.clone();
                source_indices[dim] = source_dim_idx;

                let source_strides = compute_strides(tensors[source_tensor_idx].shape().dims());
                let source_idx = ravel_index(
                    &source_indices,
                    &source_strides,
                    tensors[source_tensor_idx].shape().dims(),
                );

                out_data.push(tensors[source_tensor_idx].data()[source_idx]);
            }
        }

        Ok(Tensor {
            data: out_data,
            shape: Shape::new(out_shape),
        })
    }

    /// Negate all elements
    pub fn neg(&self) -> Tensor {
        let data = self.data.iter().map(|x| -x).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
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

    pub fn new(dims: &[usize], data: Vec<f32>) -> Self {
        Self {
            data,
            shape: Shape::new(dims.to_vec()),
        }
    }

    pub fn zero(dims: &[usize]) -> Self {
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

    pub fn mask(&self, other: &Tensor) -> Tensor {
        let data: Vec<f32> = self
            .data()
            .iter()
            .zip(other.data().iter())
            .map(|(x, y)| if *x != 0.0 || *y != 0.0 { 1.0 } else { 0.0 })
            .collect();

        Tensor {
            data,
            shape: self.shape().clone(),
        }
    }

    pub fn masked_fill(&self, mask: &Tensor, value: f32) -> Result<Self, TensorError> {
        if self.shape == mask.shape {
            // Direct element-wise operation when shapes match
            let data = self
                .data
                .iter()
                .zip(mask.data.iter())
                .map(|(x, m)| if *m != 0.0 { value } else { *x })
                .collect();
            Ok(Tensor {
                data,
                shape: self.shape.clone(),
            })
        } else {
            // Use broadcasting when shapes don't match
            let result_shape = self.shape.broadcast(&mask.shape)?;

            // Ensure the result shape matches the larger tensor (self)
            let final_shape = if result_shape.dims() == self.shape.dims() {
                self.shape.clone()
            } else {
                result_shape
            };

            let mut data = Vec::with_capacity(final_shape.num_elements());

            // Get dimensions for easier broadcasting
            let self_dims = self.shape.dims();
            let mask_dims = mask.shape.dims();
            let out_dims = final_shape.dims();

            // Calculate strides for efficient indexing
            let self_strides = compute_strides(self_dims);
            let mask_strides = compute_strides(mask_dims);
            let out_strides = compute_strides(out_dims);

            for out_idx in 0..final_shape.num_elements() {
                // Convert flat index to multi-dimensional indices
                let out_indices = unravel_index(out_idx, &out_strides, out_dims);

                // Map to self indices (handle broadcasting)
                let mut self_indices = Vec::with_capacity(self_dims.len());
                let offset = out_dims.len() - self_dims.len();
                for i in 0..self_dims.len() {
                    let out_idx = if i + offset < out_indices.len() {
                        out_indices[i + offset]
                    } else {
                        0
                    };
                    // Handle size-1 dimensions (broadcasting)
                    if self_dims[i] == 1 {
                        self_indices.push(0);
                    } else {
                        self_indices.push(out_idx);
                    }
                }

                // Map to mask indices (handle broadcasting)
                let mut mask_indices = Vec::with_capacity(mask_dims.len());
                let mask_offset = out_dims.len() - mask_dims.len();
                for i in 0..mask_dims.len() {
                    let out_idx = if i + mask_offset < out_indices.len() {
                        out_indices[i + mask_offset]
                    } else {
                        0
                    };
                    // Handle size-1 dimensions (broadcasting)
                    if mask_dims[i] == 1 {
                        mask_indices.push(0);
                    } else {
                        mask_indices.push(out_idx);
                    }
                }

                // Get flat indices
                let self_flat_idx = ravel_index(&self_indices, &self_strides, self_dims);
                let mask_flat_idx = ravel_index(&mask_indices, &mask_strides, mask_dims);

                // Apply mask
                let self_val = self.data[self_flat_idx];
                let mask_val = mask.data[mask_flat_idx];

                data.push(if mask_val != 0.0 { value } else { self_val });
            }

            Ok(Tensor {
                data,
                shape: final_shape,
            })
        }
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

    pub fn powf(&self, exp: f32) -> Self {
        let data = self.data.iter().map(|x| x.powf(exp)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn mean(&self, axis: Option<usize>, keepdim: bool) -> Result<Self, TensorError> {
        let shape = &self.shape.0;

        match axis {
            None => {
                // Mean of all elements
                let sum: f32 = self.data.iter().sum();
                let count = self.data.len() as f32;
                let result_shape = if keepdim {
                    vec![1; shape.len()] // Keep all dimensions as 1
                } else {
                    vec![1] // Scalar
                };
                Ok(Tensor {
                    data: vec![sum / count],
                    shape: Shape::new(result_shape),
                })
            }
            Some(axis) => {
                if axis >= shape.len() {
                    return Err(TensorError::DimensionMismatch {
                        expected: (shape.len(), axis),
                        found: (shape.len(), axis),
                    });
                }

                // Create output shape
                let mut out_shape = shape.clone();
                let axis_size = out_shape[axis];

                if keepdim {
                    out_shape[axis] = 1; // Keep dimension as 1
                } else {
                    out_shape.remove(axis); // Remove dimension
                    if out_shape.is_empty() {
                        out_shape = vec![1]; // Keep at least 1D
                    }
                }

                let out_size = out_shape.iter().product::<usize>();
                let mut out_data = vec![0.0; out_size];

                let strides = compute_strides(shape);
                let out_strides = compute_strides(&out_shape);

                // Accumulate values
                for idx in 0..self.data.len() {
                    let indices = unravel_index(idx, &strides, shape);

                    let mut out_indices = indices.clone();
                    if keepdim {
                        out_indices[axis] = 0; // Map to 0 in the reduced dimension
                    } else {
                        out_indices.remove(axis);
                    }

                    let out_idx = if out_indices.is_empty() {
                        0
                    } else {
                        ravel_index(&out_indices, &out_strides, &out_shape)
                    };

                    out_data[out_idx] += self.data[idx];
                }

                // Divide by axis size to get mean
                for val in &mut out_data {
                    *val /= axis_size as f32;
                }

                Ok(Tensor {
                    data: out_data,
                    shape: Shape::new(out_shape),
                })
            }
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
        let total = dims.iter().product(); //TODO: maybe this isn't correct after all
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

        // Parallelize the main loop
        new_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(new_idx, new_val)| {
                // Convert new_idx back to multi-dimensional indices
                let new_indices = unravel_index(new_idx, &new_strides, &new_shape);

                // Swap dimensions to get old indices
                let mut old_indices = new_indices.clone();
                old_indices.swap(dim0, dim1);

                // Convert back to flat index
                let old_idx = ravel_index(&old_indices, &old_strides, &self.shape.0);
                *new_val = self.data[old_idx];
            });

        Tensor {
            data: new_data,
            shape: Shape::new(new_shape),
        }
    }

    pub fn transpose_chunked(&self, dim0: usize, dim1: usize) -> Tensor {
        let mut new_shape = self.shape.0.clone();
        new_shape.swap(dim0, dim1);
        let old_strides = compute_strides(&self.shape.0);
        let new_strides = compute_strides(&new_shape);

        let new_data: Vec<f32> = (0..self.data.len())
            .into_par_iter()
            .map(|new_idx| {
                let new_indices = unravel_index(new_idx, &new_strides, &new_shape);
                let mut old_indices = new_indices.clone();
                old_indices.swap(dim0, dim1);
                let old_idx = ravel_index(&old_indices, &old_strides, &self.shape.0);
                self.data[old_idx]
            })
            .collect();

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
    /// Optimized with parallelization and cache-friendly memory access patterns
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        let a_shape = &self.shape.0;
        let b_shape = &other.shape.0;
        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();

        match (a_ndim, b_ndim) {
            // Case 1: Both 1D - dot product, result is scalar
            (1, 1) => {
                if a_shape[0] != b_shape[0] {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: a_shape[0],
                        right_rows: b_shape[0],
                    });
                }

                // Parallel dot product with chunk-based reduction
                let dot: f32 = self
                    .data
                    .par_iter()
                    .zip(other.data.par_iter())
                    .map(|(a, b)| a * b)
                    .sum();

                Ok(Tensor {
                    data: vec![dot],
                    shape: Shape::new(vec![]), // Scalar
                })
            }

            // Case 2: 1D × 2D - optimized with parallelization
            (1, 2) => {
                if a_shape[0] != b_shape[0] {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: a_shape[0],
                        right_rows: b_shape[0],
                    });
                }

                let (k, n) = (b_shape[0], b_shape[1]);

                // Parallel computation over output columns
                let data: Vec<f32> = (0..n)
                    .into_par_iter()
                    .map(|j| {
                        let mut sum = 0.0;
                        for k_idx in 0..k {
                            sum += self.data[k_idx] * other.data[k_idx * n + j];
                        }
                        sum
                    })
                    .collect();

                Ok(Tensor {
                    data,
                    shape: Shape::new(vec![n]),
                })
            }

            // Case 3: 2D × 1D - optimized with parallelization
            (2, 1) => {
                let (m, k) = (a_shape[0], a_shape[1]);
                if k != b_shape[0] {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: k,
                        right_rows: b_shape[0],
                    });
                }

                // Parallel computation over output rows
                let data: Vec<f32> = (0..m)
                    .into_par_iter()
                    .map(|i| {
                        let mut sum = 0.0;
                        let row_start = i * k;
                        for k_idx in 0..k {
                            sum += self.data[row_start + k_idx] * other.data[k_idx];
                        }
                        sum
                    })
                    .collect();

                Ok(Tensor {
                    data,
                    shape: Shape::new(vec![m]),
                })
            }

            // Case 4: 2D × 2D - highly optimized matrix multiplication
            (2, 2) => {
                let (m, k1) = (a_shape[0], a_shape[1]);
                let (k2, n) = (b_shape[0], b_shape[1]);
                if k1 != k2 {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: k1,
                        right_rows: k2,
                    });
                }

                // Choose algorithm based on matrix size
                if m * k1 * n < 8192 {
                    // Small matrices: simple parallel row-wise computation
                    self.matmul_2d_small(other, m, k1, n)
                } else if m >= 64 && n >= 64 && k1 >= 64 {
                    // Large matrices: blocked algorithm with parallelization
                    self.matmul_2d_blocked(other, m, k1, n)
                } else {
                    // Medium matrices: cache-friendly row-parallel
                    self.matmul_2d_medium(other, m, k1, n)
                }
            }

            // Case 5: 1D × ND (N > 2) - parallel batched computation
            (1, b_ndim) if b_ndim > 2 => {
                let k = a_shape[0];
                if k != b_shape[b_ndim - 2] {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: k,
                        right_rows: b_shape[b_ndim - 2],
                    });
                }

                let n = b_shape[b_ndim - 1];
                let batch_dims = &b_shape[..b_ndim - 2];
                let batch_size: usize = batch_dims.iter().product();

                let data: Vec<f32> = (0..batch_size)
                    .into_par_iter()
                    .flat_map(|batch_idx| {
                        let b_batch_offset = batch_idx * k * n;
                        (0..n)
                            .map(move |j| {
                                let mut sum = 0.0;
                                for k_idx in 0..k {
                                    sum += self.data[k_idx]
                                        * other.data[b_batch_offset + k_idx * n + j];
                                }
                                sum
                            })
                            .collect::<Vec<f32>>()
                    })
                    .collect();

                let mut output_shape = batch_dims.to_vec();
                output_shape.push(n);
                Ok(Tensor {
                    data,
                    shape: Shape::new(output_shape),
                })
            }

            // Case 6: ND × 1D (N > 2) - parallel batched computation
            (a_ndim, 1) if a_ndim > 2 => {
                let k = b_shape[0];
                let m = a_shape[a_ndim - 2];
                if a_shape[a_ndim - 1] != k {
                    return Err(TensorError::InvalidMatrixMultiplication {
                        left_cols: a_shape[a_ndim - 1],
                        right_rows: k,
                    });
                }

                let batch_dims = &a_shape[..a_ndim - 2];
                let batch_size: usize = batch_dims.iter().product();

                use rayon::prelude::*;
                let data: Vec<f32> = (0..batch_size)
                    .into_par_iter()
                    .flat_map(|batch_idx| {
                        let a_batch_offset = batch_idx * m * k;
                        (0..m)
                            .map(move |i| {
                                let mut sum = 0.0;
                                let row_start = a_batch_offset + i * k;
                                for k_idx in 0..k {
                                    sum += self.data[row_start + k_idx] * other.data[k_idx];
                                }
                                sum
                            })
                            .collect::<Vec<f32>>()
                    })
                    .collect();

                Ok(Tensor {
                    data,
                    shape: Shape::new(batch_dims.to_vec()),
                })
            }

            // Case 7: General ND × MD batched matmul - highly optimized
            (a_ndim, b_ndim) if a_ndim >= 2 && b_ndim >= 2 => {
                self.matmul_batched_optimized(other, a_ndim, b_ndim)
            }

            // Invalid cases
            _ => Err(TensorError::InvalidMatrixMultiplication {
                left_cols: *a_shape.last().unwrap_or(&0),
                right_rows: *b_shape.first().unwrap_or(&0),
            }),
        }
    }

    // Helper method for small 2D matrices
    fn matmul_2d_small(
        &self,
        other: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor, TensorError> {

        let data: Vec<f32> = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                (0..n)
                    .map(move |j| {
                        let mut sum = 0.0;
                        let row_start = i * k;
                        for k_idx in 0..k {
                            sum += self.data[row_start + k_idx] * other.data[k_idx * n + j];
                        }
                        sum
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        Ok(Tensor {
            data,
            shape: Shape::new(vec![m, n]),
        })
    }

    // Helper method for medium 2D matrices
    fn matmul_2d_medium(
        &self,
        other: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor, TensorError> {
        use rayon::prelude::*;

        let mut data = vec![0.0; m * n];

        // Parallel over rows with better cache utilization
        data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let row_start = i * k;
            for j in 0..n {
                let mut sum = 0.0;
                // Inner loop optimized for cache
                for k_idx in 0..k {
                    sum += self.data[row_start + k_idx] * other.data[k_idx * n + j];
                }
                row[j] = sum;
            }
        });

        Ok(Tensor {
            data,
            shape: Shape::new(vec![m, n]),
        })
    }

    // Helper method for large 2D matrices using blocked algorithm
    fn matmul_2d_blocked(
        &self,
        other: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor, TensorError> {
        const BLOCK_SIZE: usize = 64;
        let mut data = vec![0.0; m * n];

        // Process blocks of rows in parallel
        data.par_chunks_mut(BLOCK_SIZE * n)
            .enumerate()
            .for_each(|(block_idx, chunk)| {
                let i_start = block_idx * BLOCK_SIZE;
                let i_end = (i_start + BLOCK_SIZE).min(m);
                let actual_rows = i_end - i_start;

                for j_start in (0..n).step_by(BLOCK_SIZE) {
                    let j_end = (j_start + BLOCK_SIZE).min(n);

                    for k_start in (0..k).step_by(BLOCK_SIZE) {
                        let k_end = (k_start + BLOCK_SIZE).min(k);

                        // Process block with optimal memory access pattern
                        for i_rel in 0..actual_rows {
                            let i_abs = i_start + i_rel;
                            let out_row_start = i_rel * n;
                            let a_row_start = i_abs * k;

                            for k_idx in k_start..k_end {
                                let a_val = self.data[a_row_start + k_idx];
                                let b_row_start = k_idx * n;

                                for j in j_start..j_end {
                                    chunk[out_row_start + j] += a_val * other.data[b_row_start + j];
                                }
                            }
                        }
                    }
                }
            });

        Ok(Tensor {
            data,
            shape: Shape::new(vec![m, n]),
        })
    }

    // Helper method for batched matrix multiplication
    fn matmul_batched_optimized(
        &self,
        other: &Tensor,
        a_ndim: usize,
        b_ndim: usize,
    ) -> Result<Tensor, TensorError> {
        let a_shape = &self.shape.0;
        let b_shape = &other.shape.0;

        // Get matrix dimensions (last 2 dimensions)
        let (m, k1) = (a_shape[a_ndim - 2], a_shape[a_ndim - 1]);
        let (k2, n) = (b_shape[b_ndim - 2], b_shape[b_ndim - 1]);

        if k1 != k2 {
            return Err(TensorError::InvalidMatrixMultiplication {
                left_cols: k1,
                right_rows: k2,
            });
        }

        // Broadcast batch dimensions
        let a_batch_dims = &a_shape[..a_ndim - 2];
        let b_batch_dims = &b_shape[..b_ndim - 2];

        let batch_shape = if a_batch_dims.is_empty() && b_batch_dims.is_empty() {
            vec![]
        } else if a_batch_dims.is_empty() {
            b_batch_dims.to_vec()
        } else if b_batch_dims.is_empty() {
            a_batch_dims.to_vec()
        } else {
            Shape::new(a_batch_dims.to_vec())
                .broadcast(&Shape::new(b_batch_dims.to_vec()))?
                .0
        };

        let batch_size = if batch_shape.is_empty() {
            1
        } else {
            batch_shape.iter().product()
        };
        let a_batch_size: usize = if a_batch_dims.is_empty() {
            1
        } else {
            a_batch_dims.iter().product()
        };
        let b_batch_size: usize = if b_batch_dims.is_empty() {
            1
        } else {
            b_batch_dims.iter().product()
        };

        use rayon::prelude::*;

        // Parallel batched computation with optimized inner loops
        let data: Vec<f32> = (0..batch_size)
            .into_par_iter()
            .flat_map(|batch_idx| {
                // Handle broadcasting for batch indices
                let a_batch_idx = if a_batch_size == 1 {
                    0
                } else if a_batch_size == batch_size {
                    batch_idx
                } else {
                    batch_idx % a_batch_size
                };

                let b_batch_idx = if b_batch_size == 1 {
                    0
                } else if b_batch_size == batch_size {
                    batch_idx
                } else {
                    batch_idx % b_batch_size
                };

                let a_batch_offset = a_batch_idx * m * k1;
                let b_batch_offset = b_batch_idx * k2 * n;

                // Optimized 2D matrix multiplication for this batch
                let mut batch_result = vec![0.0; m * n];

                if m * k1 * n < 4096 {
                    // Small batch matrices - simple algorithm
                    for i in 0..m {
                        let a_row_start = a_batch_offset + i * k1;
                        let out_row_start = i * n;

                        for j in 0..n {
                            let mut sum = 0.0;
                            for k_idx in 0..k1 {
                                sum += self.data[a_row_start + k_idx]
                                    * other.data[b_batch_offset + k_idx * n + j];
                            }
                            batch_result[out_row_start + j] = sum;
                        }
                    }
                } else {
                    // Larger batch matrices - cache-friendly blocked approach
                    const BATCH_BLOCK_SIZE: usize = 32;

                    for i_start in (0..m).step_by(BATCH_BLOCK_SIZE) {
                        let i_end = (i_start + BATCH_BLOCK_SIZE).min(m);

                        for j_start in (0..n).step_by(BATCH_BLOCK_SIZE) {
                            let j_end = (j_start + BATCH_BLOCK_SIZE).min(n);

                            for k_start in (0..k1).step_by(BATCH_BLOCK_SIZE) {
                                let k_end = (k_start + BATCH_BLOCK_SIZE).min(k1);

                                for i in i_start..i_end {
                                    let a_row_start = a_batch_offset + i * k1;
                                    let out_row_start = i * n;

                                    for k_idx in k_start..k_end {
                                        let a_val = self.data[a_row_start + k_idx];
                                        let b_row_start = b_batch_offset + k_idx * n;

                                        for j in j_start..j_end {
                                            batch_result[out_row_start + j] +=
                                                a_val * other.data[b_row_start + j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                batch_result
            })
            .collect();

        // Construct output shape: batch_dims + [m, n]
        let mut output_shape = batch_shape;
        output_shape.extend_from_slice(&[m, n]);

        Ok(Tensor {
            data,
            shape: Shape::new(output_shape),
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_tensor() {
        let a = Tensor::zero(&[3, 2]);
        assert_eq!(a.shape(), &Shape::new(vec![3, 2]));
        assert_eq!(a.data(), &[0.0; 6]);
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

        assert_eq!(a.mean(None, false).unwrap(), Tensor::from([2.5]));
    }

    #[test]
    fn test_mean_axis() -> Result<(), TensorError> {
        let a = Tensor::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
        // Shape: [2, 2, 2]

        // Mean along axis 0: [2, 2]
        let result = a.mean(Some(0), false)?;
        let expected = Tensor::from([[3.0, 4.0], [5.0, 6.0]]);
        assert_eq!(result, expected);

        // Mean along axis 1: [2, 2]
        let result = a.mean(Some(1), false)?;
        let expected = Tensor::from([[2.0, 3.0], [6.0, 7.0]]);
        assert_eq!(result, expected);

        // Mean along axis 2: [2, 2]
        let result = a.mean(Some(2), false)?;
        let expected = Tensor::from([[1.5, 3.5], [5.5, 7.5]]);
        assert_eq!(result, expected);

        // Mean of all elements: scalar
        let result = a.mean(None, false)?;
        let expected = Tensor::from([4.5]);
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_mean_keepdim() -> Result<(), TensorError> {
        let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

        // Mean along axis 0 with keepdim
        let result = a.mean(Some(0), true)?;
        let expected = Tensor::from([[2.0, 3.0]]); // Shape: [1, 2]
        assert_eq!(result, expected);

        // Mean along axis 1 with keepdim
        let result = a.mean(Some(1), true)?;
        let expected = Tensor::from([[1.5], [3.5]]); // Shape: [2, 1]
        assert_eq!(result, expected);

        Ok(())
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
        let expected = Tensor::from([[1.0, -1.0], [-1.0, 4.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_masked_fill_basic() {
        // Basic masked_fill with boolean-like mask
        let scores = Tensor::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let mask = Tensor::from([
            [1.0, 0.0, 1.0], // True, False, True
            [0.0, 1.0, 0.0], // False, True, False
            [1.0, 0.0, 1.0], // True, False, True
        ]);

        // Apply masked_fill: where mask is non-zero, fill with f32::NEG_INFINITY
        let result = scores.masked_fill(&mask, f32::NEG_INFINITY).unwrap();
        let expected = Tensor::from([
            [f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY],
            [4.0, f32::NEG_INFINITY, 6.0],
            [f32::NEG_INFINITY, 8.0, f32::NEG_INFINITY],
        ]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_masked_fill_causal_attention() {
        // Causal attention masking (from PyTorch notebook)
        let attn_scores = Tensor::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        // Create causal mask (upper triangular) - masks future tokens
        let ones = Tensor::ones(&[3, 3]);
        let mask_global = ones.triu(1); // diagonal=1 means mask upper triangle

        let masked_scores = attn_scores
            .masked_fill(&mask_global, f32::NEG_INFINITY)
            .unwrap();
        let expected = Tensor::from([
            [1.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            [4.0, 5.0, f32::NEG_INFINITY],
            [7.0, 8.0, 9.0],
        ]);

        assert_eq!(masked_scores, expected);
    }

    #[test]
    fn test_masked_fill_sliding_window() {
        // Sliding window attention
        let sliding_window = 2;
        let seq_len = 5;

        let ones = Tensor::ones(&[seq_len, seq_len]);

        // Future mask (causal)
        let mask_global = ones.triu(1);

        // Far past mask (beyond sliding window)
        // This creates a mask for positions too far in the past
        let far_past = ones.triu(sliding_window as isize).transpose(0, 1);

        // Combined local mask using logical OR
        let mask_local = &mask_global.mask(&far_past); // Using your mask function as OR

        let scores = Tensor::from([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ]);

        let masked_scores = scores.masked_fill(&mask_local, f32::NEG_INFINITY).unwrap();

        // Verify the pattern: each position can only attend to recent past within window
        assert_eq!(masked_scores.data()[0], 1.0); // [0,0] - can attend to self
        assert_eq!(masked_scores.data()[1], f32::NEG_INFINITY); // [0,1] - future masked
        assert_eq!(masked_scores.data()[6], 7.0); // [1,1] - can attend to self
        assert_eq!(masked_scores.data()[7], f32::NEG_INFINITY); // [1,2] - future masked
    }

    #[test]
    fn test_masked_fill_different_values() {
        // Fill with different values
        let data = Tensor::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let mask = Tensor::from([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]);

        // Fill with zero
        let result1 = data.masked_fill(&mask, 0.0).unwrap();
        let expected1 = Tensor::from([[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]]);
        assert_eq!(result1, expected1);

        // Fill with -999
        let result2 = data.masked_fill(&mask, -999.0).unwrap();
        let expected2 = Tensor::from([[-999.0, 2.0, -999.0], [4.0, -999.0, 6.0]]);
        assert_eq!(result2, expected2);

        // Fill with NaN
        let result3 = data.masked_fill(&mask, f32::NAN).unwrap();
        // For NaN comparison, check each element individually
        assert!(result3.data()[0].is_nan());
        assert_eq!(result3.data()[1], 2.0);
        assert!(result3.data()[2].is_nan());
        assert_eq!(result3.data()[3], 4.0);
        assert!(result3.data()[4].is_nan());
        assert_eq!(result3.data()[5], 6.0);
    }

    #[test]
    fn test_masked_fill_broadcasting() {
        // Broadcasting with different shapes (like attention scores)
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 3;

        // Attention scores: [batch, heads, seq_len, seq_len]
        let scores = Tensor::ones(&[batch_size, num_heads, seq_len, seq_len]);

        // Mask is only [seq_len, seq_len] - should be broadcasted
        let mask = Tensor::from([
            [0.0, 1.0, 1.0], // Can attend to position 0, mask 1,2
            [0.0, 0.0, 1.0], // Can attend to 0,1, mask 2
            [0.0, 0.0, 0.0], // Can attend to 0,1,2
        ]);

        let masked_scores = scores.masked_fill(&mask, f32::NEG_INFINITY).unwrap();

        let expecterd = Tensor::from([
            [
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
            ],
            [
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
                [
                    [1., f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [1., 1., f32::NEG_INFINITY],
                    [1., 1., 1.],
                ],
            ],
        ]);
        assert_eq!(masked_scores, expecterd);
    }

    #[test]
    fn test_masked_fill_all_true_mask() {
        let all_true_mask = Tensor::ones(&[3, 3]);

        let attn_scores = Tensor::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let masked_scores = attn_scores
            .masked_fill(&all_true_mask, f32::NEG_INFINITY)
            .unwrap();

        let expected = Tensor::from([
            [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        ]);
        assert_eq!(masked_scores, expected);
    }

    #[test]
    fn test_masked_fill_all_false_mask() {
        // Test with all False mask (all zeros)
        let all_false_mask = Tensor::zero(&[3, 3]);

        let attn_scores = Tensor::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        let masked_scores = attn_scores
            .masked_fill(&all_false_mask, f32::NEG_INFINITY)
            .unwrap();

        // All values should remain unchanged since mask is all False
        assert_eq!(masked_scores, attn_scores);
    }

    #[test]
    fn test_masked_fill_shape_mismatch_error() {
        let scores = Tensor::ones(&[2, 3]);
        let wrong_mask = Tensor::ones(&[3, 4]); // Wrong shape

        let result = scores.masked_fill(&wrong_mask, -1.0);
        assert!(result.is_err()); // Should fail due to incompatible shapes
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
