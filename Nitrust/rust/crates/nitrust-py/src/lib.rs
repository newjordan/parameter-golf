use nitrust_mmap_loader::MmapTokenLoader;
use nitrust_pinned_batcher::{build_language_model_batch, BatchSpec};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyfunction]
fn mmap_read_tokens(path: String, token_offset: usize, token_len: usize) -> PyResult<Vec<u16>> {
    let loader =
        MmapTokenLoader::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = loader
        .read_tokens(token_offset, token_len)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(out)
}

#[pyfunction]
fn build_lm_batch(tokens: Vec<u16>, seq_len: usize, batch_size: usize) -> PyResult<(Vec<u16>, Vec<u16>)> {
    let spec = BatchSpec { seq_len, batch_size };
    build_language_model_batch(&tokens, spec).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn nitrust_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mmap_read_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(build_lm_batch, m)?)?;
    Ok(())
}
