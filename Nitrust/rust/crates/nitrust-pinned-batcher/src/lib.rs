use std::ffi::c_void;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BatcherError {
    #[error("invalid batch spec (seq_len={seq_len}, batch_size={batch_size})")]
    InvalidBatchSpec { seq_len: usize, batch_size: usize },
    #[error("not enough tokens ({provided}) for requested batch ({required})")]
    NotEnoughTokens { provided: usize, required: usize },
    #[error("failed to page-lock host buffer: {0}")]
    MlockFailed(std::io::Error),
}

#[derive(Clone, Copy, Debug)]
pub struct BatchSpec {
    pub seq_len: usize,
    pub batch_size: usize,
}

impl BatchSpec {
    pub fn validate(self) -> Result<Self, BatcherError> {
        if self.seq_len == 0 || self.batch_size == 0 {
            return Err(BatcherError::InvalidBatchSpec {
                seq_len: self.seq_len,
                batch_size: self.batch_size,
            });
        }
        Ok(self)
    }

    pub fn batch_tokens(self) -> usize {
        self.seq_len * self.batch_size
    }
}

/// CPU-resident page-locked buffer for staging host-side token data.
///
/// This is a generic OS page-lock (`mlock`) primitive. CUDA pinned-memory
/// registration can be layered on top in a later module.
pub struct PinnedU16Buffer {
    data: Vec<u16>,
    is_pinned: bool,
}

impl PinnedU16Buffer {
    pub fn try_new(len: usize) -> Result<Self, BatcherError> {
        let mut data = vec![0u16; len];
        if len == 0 {
            return Ok(Self {
                data,
                is_pinned: false,
            });
        }
        let bytes = len * std::mem::size_of::<u16>();
        let ptr = data.as_mut_ptr().cast::<c_void>();
        let rc = unsafe { libc::mlock(ptr, bytes) };
        if rc != 0 {
            return Err(BatcherError::MlockFailed(std::io::Error::last_os_error()));
        }
        Ok(Self {
            data,
            is_pinned: true,
        })
    }

    pub fn new_best_effort(len: usize) -> Self {
        match Self::try_new(len) {
            Ok(buf) => buf,
            Err(_) => Self {
                data: vec![0u16; len],
                is_pinned: false,
            },
        }
    }

    pub fn as_slice(&self) -> &[u16] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u16] {
        &mut self.data
    }

    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }
}

impl Drop for PinnedU16Buffer {
    fn drop(&mut self) {
        if self.is_pinned && !self.data.is_empty() {
            let bytes = self.data.len() * std::mem::size_of::<u16>();
            let ptr = self.data.as_mut_ptr().cast::<c_void>();
            let _ = unsafe { libc::munlock(ptr, bytes) };
        }
    }
}

/// Build contiguous `(x, y)` language-model batches from a token stream.
///
/// - `x`: input tokens
/// - `y`: next-token labels (`x` shifted by +1)
pub fn build_language_model_batch(tokens: &[u16], spec: BatchSpec) -> Result<(Vec<u16>, Vec<u16>), BatcherError> {
    let spec = spec.validate()?;
    let batch_tokens = spec.batch_tokens();
    let required = batch_tokens + 1;
    if tokens.len() < required {
        return Err(BatcherError::NotEnoughTokens {
            provided: tokens.len(),
            required,
        });
    }
    let mut x = vec![0u16; batch_tokens];
    let mut y = vec![0u16; batch_tokens];
    x.copy_from_slice(&tokens[..batch_tokens]);
    y.copy_from_slice(&tokens[1..required]);
    Ok((x, y))
}

#[cfg(test)]
mod tests {
    use super::{build_language_model_batch, BatchSpec, PinnedU16Buffer};

    #[test]
    fn builds_shifted_batch() {
        let tokens: Vec<u16> = (0..17).collect();
        let spec = BatchSpec {
            seq_len: 4,
            batch_size: 4,
        };
        let (x, y) = build_language_model_batch(&tokens, spec).expect("batch");
        assert_eq!(x.len(), 16);
        assert_eq!(y.len(), 16);
        assert_eq!(x[0], 0);
        assert_eq!(x[15], 15);
        assert_eq!(y[0], 1);
        assert_eq!(y[15], 16);
    }

    #[test]
    fn best_effort_buffer_allocates() {
        let mut buf = PinnedU16Buffer::new_best_effort(8);
        let s = buf.as_mut_slice();
        s[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }
}
