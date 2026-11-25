# Parameter Golf

Parameter Golf is a challenge to train a language model that fits in a 16 MB artifact and trains in under 10 minutes on 8xH100s. The score is tokenizer-agnostic validation bits per byte (`val_bpb`) on FineWeb. Lower is better.

The published challenge window was March 18, 2026 through April 30, 2026. This repository keeps the starter scripts, the checked-in submissions under `records/`, and a static research hub that ranks those submissions.

The challenge is inspired by [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). Attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The project license is the MIT license in [LICENSE](LICENSE).

## Repository map

| Path | What it is |
| --- | --- |
| `train_gpt.py` | CUDA starter. Configuration is environment variables. `--help` exits before PyTorch is imported. |
| `train_gpt_mlx.py` | Apple Silicon starter. `--help` exits before MLX is imported. Tied embeddings only. |
| `data/` | FineWeb download and tokenizer export helpers. See [data/README.md](data/README.md). |
| `records/` | Checked-in submissions. Each record is a folder with a README, `submission.json`, a train log, and a training script. |
| `index.html`, `app.js`, `hub/view_model.js`, `styles.css` | Static research hub. |
| `hub_index.json` | Generated index of `records/`. The hub loads this file. |
| `parameter_golf/` | Index builder, shard writer, and the budget checks covered by the tests. The starter scripts do not import this package, so a copied submission can still run from its own folder. |
| `tests/` | Pytest suite and Node tests for the hub view model. |
| `.github/workflows/ci.yml` | GitHub Actions workflow. |

Both starter scripts are required to stay under 1500 lines. `train_gpt.py` is 1190 lines and `train_gpt_mlx.py` is 1182 lines.

## Research hub

Open `index.html` in a browser, or serve the repository root as static files. [vercel.json](vercel.json) deploys the repository root with no build step. Browsing the hub does not need Python.

The page ranks every `submission.json`, shows the delta versus the Naive Baseline, and opens the log or writeup for a selected row. File sizes use decimal units: 1 MB is 1,000,000 bytes, and the artifact cap is 16,000,000 bytes. Search covers path, author, track, and snippet. Record rows are buttons and respond to Enter and Space.

Rebuild the index after changing anything under `records/`:

```bash
python -m parameter_golf.build_index
python -m parameter_golf.build_index --check
```

The current index was built from `records/` only. It contains 120 records and 23 submissions. `generated_at` is `2026-03-23T00:00:00Z`, the newest submission date in those files. Category counts are 23 submissions, 23 reports, 23 scripts, 50 run logs, and 1 TSV. Status counts are 23 `ok` and 97 `unknown`. The builder found 73 records with a numeric metric, 22 ablation deltas against the Naive Baseline, and 21 submissions on the 10-minute track.

The index skips dotenv files, private-key filenames, and binary artifacts. Snippets redact credential assignments, SSH URLs, and PEM private-key blocks. Paths in the index are relative to the repository.

## Checked-in scores

These values are the `val_bpb` and artifact byte fields stored in each `submission.json`. The hub reads `bytes_total`, `artifact_bytes`, `artifact_size_bytes`, or `artifact_bytes_mean`. Every listed artifact is between 15,327,112 and 15,990,006 bytes, under the 16,000,000 byte cap.

The lowest 10-minute `val_bpb` is 1.1194 (`LeakyReLU² + Legal Score-First TTT + Parallel Muon`). The smallest artifact is the non-record SwiGLU run at 15,327,112 bytes.

An earlier copy of this README rounded some scores. LoRA TTT was listed as 1.1928; `submission.json` stores `val_bpb` 1.1929. The Int5 submission stored the published mean only as `val_loss` 1.14276, and its blurb says that number is the mean of 3 seeds. `val_bpb` is now set to the same 1.14276. Warmdown Quantization and Lower LR are checked in and were not on that earlier table.

### 10-minute track

| val_bpb | Bytes | GitHub id | Run | Record |
| ---: | ---: | --- | --- | --- |
| 1.1194 | 15990006 | abaybektursun | LeakyReLU² + Legal Score-First TTT + Parallel Muon | [info](records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md) |
| 1.12278022 | 15555017 | signalrush | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 | [info](records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md) |
| 1.12484502 | 15612308 | jfprincz | 11L Partial RoPE + LN Scale + EMA + XSA4 | [info](records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md) |
| 1.12707468 | 15534645 | jfprincz | 11L XSA + EMA + Int6 MLP3x + WD=0.04 | [info](records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md) |
| 1.13071416 | 15892986 | unnir | 11L + Efficient Partial XSA + FA3 + SWA/120 | [info](records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md) |
| 1.14276 | 15900000 | thwu1 | 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04 | [info](records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) |
| 1.14581692 | 15862650 | raahilshah | Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA | [info](records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md) |
| 1.15015359 | 15427455 | aruniyer | 11L MLP3x + Int6 QAT + zstd-22 + sliding window | [info](records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md) |
| 1.1556 | 15878809 | not named in submission.json | SmearGate + OrthoInit + Muon WD | [info](records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md) |
| 1.1574404 | 15977717 | samuellarson | Int6 MLP3x Sliding Window | [info](records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md) |
| 1.15861696 | 15558319 | yahya010 | 10L Int6 QAT + Zstd MLP2.6x Muon0.99 Sliding Window | [info](records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/README.md) |
| 1.16301431 | 15353490 | aquariouseworkman | Mixed Quant + Sliding Window Eval | [info](records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md) |
| 1.17475315 | 15374243 | notapplica | Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init | [info](records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md) |
| 1.19250007 | 15874829 | mattqlf | Sliding Window Eval (stride=64) | [info](records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md) |
| 1.1929 | 15882446 | samacqua | LoRA TTT | [info](records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md) |
| 1.20143417 | 15868326 | spokane-way | Training Opt Seq4096 v1 | [info](records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md) |
| 1.20576485 | 15867270 | spokane-way | Long Context Seq2048 v2 | [info](records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md) |
| 1.214745 | 15928974 | nanlliu | 10L Mixed Precision | [info](records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md) |
| 1.21972502 | 15896222 | chonchiog | FP16 Tied Embedding + LR/Warmdown Tuning | [info](records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md) |
| 1.22296644 | 15854246 | nanlliu | Lower LR | [info](records/track_10min_16mb/2026-03-18_LowerLR/README.md) |
| 1.2243657 | 15863489 | openai | Naive Baseline | [info](records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) |

The SmearGate submission does not set `author` or `github_id`. An earlier leaderboard in this file credited that row to aquariouseworkman. That name is not in the submission file or its README.

### Non-record track

These runs are checked in under `records/track_non_record_16mb/`. They are not 10-minute leaderboard entries.

| val_bpb | Bytes | GitHub id | Run | Record |
| ---: | ---: | --- | --- | --- |
| 1.20737944 | 15810161 | williamd | 4-Hour Quasi-10B SP1024 | [info](records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md) |
| 1.32814313 | 15327112 | not named in submission.json | SwiGLU + Warmdown Fix + Quarter Batch (1x5090) | [info](records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md) |

The SwiGLU `author` and `github_id` fields are empty strings.

## Setup

Python 3.11 or newer. The CI workflow uses Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional extras in [pyproject.toml](pyproject.toml):

| Extra | Use |
| --- | --- |
| `dev` | Pytest, the index builder's Hugging Face import, and NumPy. |
| `data` | Dataset download and tokenizer export. |
| `train` | CUDA training dependencies, including PyTorch. |
| `mlx` | Apple Silicon training. MLX does not install on Linux. |

`requirements.txt` is the CUDA reference list. CI does not install it. The hub tests install `.[dev]` and do not install PyTorch.

This working tree was tested with NumPy 2.5.5, pytest 9.2.0, and huggingface-hub 2.4.0. NumPy 2 raises `ValueError` from `np.array(..., copy=False)` when a cast requires a copy. The data exporter uses an explicit contiguous cast for that reason.

## Download a dataset

The default published repo is `willdepueoai/parameter-golf`. A full download is the validation split plus 80 training shards of 100,000,000 tokens each.

```bash
python -m pip install -e ".[data]"
python3 data/cached_challenge_fineweb.py --variant sp1024
```

For a one-shard smoke download:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Pass either a positional shard count or `--train-shards`, not both. The command exits with an error if both are set. Downloaded shards, the manifest, and tokenizer files are gitignored.

## Train

Training reads environment variables. Any other command-line argument exits with status 2 before the heavy imports. Neither starter was executed as a training run in this pass.

`train_gpt.py` requires an NVIDIA GPU with CUDA. `torchrun` sets `WORLD_SIZE`, which must be 1, 2, 4, or 8. Gradient accumulation is `8 / WORLD_SIZE`. `TRAIN_BATCH_TOKENS` must be divisible by `WORLD_SIZE * grad_accum_steps * TRAIN_SEQ_LEN`. The default batch is 524288 tokens, the default sequence length is 1024, and the default wall clock is 600 seconds. `VAL_LOSS_EVERY` defaults to 1000.

```bash
python3 train_gpt.py --help
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

`train_gpt_mlx.py` trains only on Apple Silicon with MLX installed. `WORLD_SIZE` is 1 and `GRAD_ACCUM_STEPS` defaults to 8. `TRAIN_BATCH_TOKENS` must be divisible by `GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN`. Each validation microbatch uses `VAL_BATCH_SIZE / GRAD_ACCUM_STEPS` tokens, so a smoke run that sets both sizes to 8192 evaluates one sequence at a time when `TRAIN_SEQ_LEN` is 1024. `VAL_LOSS_EVERY` defaults to 0, which validates once at the end.

```bash
python3 train_gpt_mlx.py --help
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Set `MAX_WALLCLOCK_SECONDS=0` to disable the 600 second cap. The scripts print one config line before training and print `val_loss`, `val_bpb`, and the compressed artifact size at the end.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m parameter_golf.build_index --check
node --test tests/hub_view.test.js
```

The last local run reported 21 passed pytest tests and 5 passed Node tests. The Python tests cover shard round-trips, the shard-count flag conflict, blank JSONL lines, the CUDA batch-divisibility rule, secret redaction, `--help` on both starters, and a byte-for-byte match of `hub_index.json`. They do not download datasets or start training.

GitHub Actions runs those commands on `ubuntu-latest` with Python 3.12 and Node 22 for pushes and pull requests.

## Rules that the records followed

New state-of-the-art records were accepted as a pull request that only added a folder under the matching `records/` track. A record needed:

1. A `val_bpb` at least 0.005 nats below the current state of the art, with logs showing `p < 0.01`. Systems-only speed improvements could waive that gap.
2. A demonstration that `val_bpb` was still correct if the tokenizer or dataset changed.
3. A reproducible run under 10 minutes on 8xH100s. Evaluation had its own 10 minute limit. Evaluation could not read the training set, and test-time training could use a validation token only after that token had already been scored.

The artifact is code bytes plus compressed model bytes. Counted code lives in the training script. The cap is 16,000,000 bytes, not 16 MiB. The artifact has to be self-contained: no downloads or network calls during evaluation.

Each submission folder contains:

1. `README.md` explaining the run.
2. `submission.json` with the author, GitHub id, `val_bpb`, and artifact size.
3. The train log, usually an average over 3 seeds.
4. A training script that runs from that folder, plus a `requirements.txt` if it needs extra packages.

Non-record folders are for ideas that stay inside the byte cap without beating the state of the art, including unlimited-compute runs. Changes to the shared starter scripts were limited to small simplifications. The best models stay in `records/`.

## Support

The compute-grant form published for the challenge is [Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form). The participant form is the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). Questions went to the Parameter Golf channels on the [OpenAI Discord](https://discord.com/invite/openai).
