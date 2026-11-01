use anyhow::{bail, Result};

/// A dense 3-D array in Z-Y-X order, with X varying fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct Volume<T> {
    pub shape: [usize; 3],
    pub data: Vec<T>,
}

impl<T> Volume<T> {
    /// Construct a volume, validating non-zero dimensions, the shape product,
    /// and the supplied data length.
    pub fn new(shape: [usize; 3], data: Vec<T>) -> Result<Self> {
        if shape.contains(&0) {
            bail!("volume dimensions must be non-zero, got {shape:?}");
        }
        let voxels = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow::anyhow!("volume shape is too large: {shape:?}"))?;
        if data.len() != voxels {
            bail!(
                "volume data length {} does not match shape {shape:?} ({} voxels)",
                data.len(),
                voxels
            );
        }
        Ok(Self { shape, data })
    }

    /// Linear row-major index of a voxel. The caller is responsible for bounds.
    pub fn idx(&self, z: usize, y: usize, x: usize) -> usize {
        (z * self.shape[1] + y) * self.shape[2] + x
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, z: usize, y: usize, x: usize) -> Option<&T> {
        if z < self.shape[0] && y < self.shape[1] && x < self.shape[2] {
            self.data.get(self.idx(z, y, x))
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, z: usize, y: usize, x: usize) -> Option<&mut T> {
        if z < self.shape[0] && y < self.shape[1] && x < self.shape[2] {
            let i = self.idx(z, y, x);
            self.data.get_mut(i)
        } else {
            None
        }
    }
}

impl<T: Clone> Volume<T> {
    pub fn filled(shape: [usize; 3], value: T) -> Result<Self> {
        if shape.contains(&0) {
            bail!("volume dimensions must be non-zero, got {shape:?}");
        }
        let voxels = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| anyhow::anyhow!("volume shape is too large: {shape:?}"))?;
        Ok(Self {
            shape,
            data: vec![value; voxels],
        })
    }
}

/// The two fluorescence channels supplied by a loaded image stack.
#[derive(Clone, Debug)]
pub struct ImageStack {
    pub membrane: Volume<u16>,
    pub nuclei: Option<Volume<u16>>,
}
