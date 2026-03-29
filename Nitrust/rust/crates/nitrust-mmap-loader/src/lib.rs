use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use thiserror::Error;

const SHARD_HEADER_WORDS_I32: usize = 256;
const SHARD_HEADER_BYTES: usize = SHARD_HEADER_WORDS_I32 * std::mem::size_of::<i32>();
const SHARD_MAGIC: i32 = 20240520;
const SHARD_VERSION: i32 = 1;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("mapped file has odd byte length ({0}); expected packed u16 tokens")]
    InvalidByteLength(usize),
    #[error("invalid shard header (magic={magic}, version={version})")]
    InvalidHeader { magic: i32, version: i32 },
    #[error("invalid shard token count in header ({0})")]
    InvalidTokenCount(i32),
    #[error("shard size mismatch (expected={expected}, actual={actual})")]
    ShardSizeMismatch { expected: usize, actual: usize },
    #[error("token span out of bounds (offset={offset}, len={len}, total={total})")]
    OutOfBounds {
        offset: usize,
        len: usize,
        total: usize,
    },
}

/// Read-only memory-mapped token shard reader.
///
/// Supports both:
/// - Medusa/ClownCar shard format: 256 x i32 header + packed little-endian `u16` tokens
/// - Legacy/raw format: packed little-endian `u16` tokens only
#[derive(Debug)]
pub struct MmapTokenLoader {
    path: PathBuf,
    mmap: Mmap,
    total_tokens: usize,
    token_data_offset: usize,
}

impl MmapTokenLoader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() % 2 != 0 {
            return Err(LoaderError::InvalidByteLength(mmap.len()));
        }

        let (token_data_offset, total_tokens) = if mmap.len() >= SHARD_HEADER_BYTES {
            let magic = read_i32_le(&mmap, 0);
            let version = read_i32_le(&mmap, 4);

            if magic == SHARD_MAGIC || version == SHARD_VERSION {
                if magic != SHARD_MAGIC || version != SHARD_VERSION {
                    return Err(LoaderError::InvalidHeader { magic, version });
                }
                let num_tokens_i32 = read_i32_le(&mmap, 8);
                if num_tokens_i32 < 0 {
                    return Err(LoaderError::InvalidTokenCount(num_tokens_i32));
                }
                let total_tokens = num_tokens_i32 as usize;
                let expected = SHARD_HEADER_BYTES + total_tokens * 2;
                if mmap.len() != expected {
                    return Err(LoaderError::ShardSizeMismatch {
                        expected,
                        actual: mmap.len(),
                    });
                }
                (SHARD_HEADER_BYTES, total_tokens)
            } else {
                (0, mmap.len() / 2)
            }
        } else {
            (0, mmap.len() / 2)
        };

        Ok(Self {
            path: path_buf,
            mmap,
            total_tokens,
            token_data_offset,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn read_tokens(&self, token_offset: usize, token_len: usize) -> Result<Vec<u16>, LoaderError> {
        let end = token_offset.saturating_add(token_len);
        if end > self.total_tokens {
            return Err(LoaderError::OutOfBounds {
                offset: token_offset,
                len: token_len,
                total: self.total_tokens,
            });
        }
        let start_b = self.token_data_offset + token_offset * 2;
        let end_b = self.token_data_offset + end * 2;
        let bytes = &self.mmap[start_b..end_b];
        let mut out = Vec::with_capacity(token_len);
        for chunk in bytes.chunks_exact(2) {
            out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Ok(out)
    }
}

fn read_i32_le(bytes: &[u8], offset: usize) -> i32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[offset..offset + 4]);
    i32::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::MmapTokenLoader;

    fn write_headered_shard(file: &mut tempfile::NamedTempFile, values: &[u16]) {
        let mut header = [0i32; 256];
        header[0] = 20240520;
        header[1] = 1;
        header[2] = i32::try_from(values.len()).expect("values length fits i32");
        for word in header {
            file.write_all(&word.to_le_bytes()).expect("write header");
        }
        for v in values {
            file.write_all(&v.to_le_bytes()).expect("write token");
        }
    }

    #[test]
    fn reads_expected_span_raw() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmp file");
        let values: [u16; 6] = [10, 20, 30, 40, 50, 60];
        for v in values {
            tmp.write_all(&v.to_le_bytes()).expect("write token");
        }
        tmp.flush().expect("flush");

        let loader = MmapTokenLoader::open(tmp.path()).expect("open loader");
        assert_eq!(loader.total_tokens(), 6);
        let span = loader.read_tokens(2, 3).expect("read span");
        assert_eq!(span, vec![30, 40, 50]);
    }

    #[test]
    fn reads_expected_span_headered() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmp file");
        let values: [u16; 6] = [10, 20, 30, 40, 50, 60];
        write_headered_shard(&mut tmp, &values);
        tmp.flush().expect("flush");

        let loader = MmapTokenLoader::open(tmp.path()).expect("open loader");
        assert_eq!(loader.total_tokens(), 6);
        let span = loader.read_tokens(2, 3).expect("read span");
        assert_eq!(span, vec![30, 40, 50]);
    }

    #[test]
    fn rejects_oob_span() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmp file");
        write_headered_shard(&mut tmp, &[1u16]);
        tmp.flush().expect("flush");

        let loader = MmapTokenLoader::open(tmp.path()).expect("open loader");
        let err = loader.read_tokens(1, 1).expect_err("should fail");
        let msg = err.to_string();
        assert!(msg.contains("out of bounds"));
    }

    #[test]
    fn rejects_header_size_mismatch() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmp file");
        let mut header = [0i32; 256];
        header[0] = 20240520;
        header[1] = 1;
        header[2] = 3; // claims 3 tokens, only write 1
        for word in header {
            tmp.write_all(&word.to_le_bytes()).expect("write header");
        }
        tmp.write_all(&99u16.to_le_bytes()).expect("write token");
        tmp.flush().expect("flush");

        let err = MmapTokenLoader::open(tmp.path()).expect_err("should fail");
        let msg = err.to_string();
        assert!(msg.contains("size mismatch"));
    }
}
