# Nitrust Rust Sprint A Scaffold

This workspace contains the first Sprint A crates:

- `nitrust-mmap-loader`: memory-mapped token shard reader (Medusa/ClownCar headered shards + raw `u16`)
- `nitrust-pinned-batcher`: host buffer and LM batch builder
- `nitrust-py`: `pyo3` bridge exposing minimal callable APIs

## Smoke Test

```bash
cd Nitrust/rust
cargo test -p nitrust-mmap-loader -p nitrust-pinned-batcher
```

Build Python bridge and smoke-load it:

```bash
./Nitrust/scripts/build_nitrust_py.sh
```

Medusa preflight parity check:

```bash
./Nitrust/scripts/medusa_nitrust_preflight.sh
```
