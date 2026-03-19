Run01: Default configuration baseline on 8xH100.

Configuration:
- train_batch_tokens: 524288
- train_seq_len: 1024
- iterations: 20000
- warmup_steps: 20
- max_wallclock_seconds: 600
- seed: 1337

Key metrics:
- Stopped at step 1203/20000 (wallclock cap)
- step_avg: 499.12ms
- Pre-quant: val_loss=2.2681, val_bpb=1.3433
- Post int8+zlib roundtrip: val_loss=2.27034992, val_bpb=1.34462911
- Peak memory: 10239 MiB allocated, 10554 MiB reserved
- Total submission size int8+zlib: 13142805 bytes

Comparison to NaiveBaseline:
- This run: 1203 steps, val_bpb=1.3446
- NaiveBaseline: 13780 steps, val_bpb=1.2244
- Gap: +0.1202 bpb (worse)
- This run is ~11.5x slower per step (499ms vs 43ms), completing only ~8.7% as many steps

The large step_avg difference suggests this may be running on fewer GPUs or with
a different parallelism config than the baseline's 8xH100 setup.
