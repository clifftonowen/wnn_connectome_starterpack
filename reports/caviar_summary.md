# Caviar experiments: procedure, results and where to pick up

This is the entry point for future sessions. It covers how experiments were run, what was found, and the current state, as of 2026-11-08.
More detail lives in `caviar/README.md` (commands and files), `reports/caviar_next_runs.md` (the latest runs), `notebooks/02_caviar_results.ipynb` and `notebooks/03_tuning_results.ipynb`.

## Current state

- **Best public score: `ema_wd_valid_pair.zip`, 0.65527**, in 44m1s. It averages two `group_gru` models with EMA and weight decay 0.1, trained on all training sequences plus all of validation.
- **Best without validation training: `final_ema_pair.zip`, 0.65440.**
- Suggested choice for the Final Dataset: those two. One trained on validation, one not, in case the final data behaves differently.
- Zips and checkpoints are git-ignored and exist only on this machine, in `caviar_runs/`. Keep a copy.
- Deadline Nov 15, 2026, 11:59 PM UTC. 5 submissions a day. Do not share code or models with anyone. The repo must stay private.

## The competition in brief

- Predict `t0` and `t1` for each row of a sequence, one row at a time. Score: global weighted Pearson over the scored rows.
- Train: 10,607 sequences. Validation: 1,873. Each has 20,000 rows and 112 features. Rows 0 to 98 are warm-up and are never scored.
- Validation has a scoring mask (`is_scored`). The test mask is hidden.
- Scorer: Linux, 1 CPU, 16 GB, no GPU, 60 minutes for the whole test set (about 39.4 million rows), deterministic, ZIP under 20 MB.
- Only `train.parquet` and `valid.parquet` may be used for training. Training on validation is allowed.

## Procedure

### Comparing models fairly

1. **Folds by whole sequence.** No sequence is split across folds. Folds are stratified on each sequence's mean `t0` and mean absolute target, in 25 strata (`caviar/folds.py`).
2. **Round 1:** 32 models on 1,500 sequences, 3 folds of 500. Same training setup for every model.
3. **Round 2:** the best 12 on 5,000 sequences, 5 folds of 1,000. The top two got 3 seeds.
4. **Every model was tested first** (`caviar/check_models.py`): it must not see future rows, and its row-by-row output must match its whole-sequence output.
5. **Training setup**, the same for all neural models:
   - window of 1,024 rows, batch 64
   - AdamW at lr 1e-3, warm-up then cosine decay
   - loss: MSE weighted by the absolute target, with targets clipped to [-2, 2]
   - early stopping on a small dev set taken from the training sequences: 5% of them, scored on their first 5,000 rows
6. **Three scores per run:**
   - `cv`: all rows of the held-out fold
   - masked validation: the scored rows only, the number closest to the leaderboard
   - all-rows validation

### Rule for keeping a change (used for all tuning)

Run it on folds 0, 1 and 2, seed 0, against a reference run with the same seed. Keep the change only if all of these hold:
- masked validation and `cv` both rise on all 3 folds
- the mean masked gain is over 0.001

Seeds alone move scores by about 0.001, so smaller gaps count as ties.

### Testing training on validation

Validation is split into two fixed halves in file order (`train.valid_halves`): A is the first 936 sequences, B the other 937. File order is sorted `seq_ix`, which looks random.
- `--valid-train A` adds half A to training and scores only half B.
- The references are the same setup without half A, also scored on half B, after checking that each reproduces its stored full score.
- `--valid-train all` trains on all of validation. Its validation score is stored apart, as `valid_seen_*`, because it is not a fair test.

### Building and checking a submission

1. `python -m caviar.package build a.pt [b.pt] --name x` exports one ONNX file. Several models are averaged inside it.
2. `python -m caviar.package check x.zip --data full --passes 2` runs the real scorer twice over all of validation. Predictions must be identical between passes.
3. The Linux check runs the scorer in a `python:3.11-slim-bookworm` container with one CPU, on a 600,000-row subset, using `scratchpad/linux_speed.sh`. This scratch script is not in the repo. It installs onnxruntime 1.23.2, numpy 2.2.6 and pyarrow, then runs `package check --data subset --passes 2`.
4. The local "under limit" flag uses this machine's speed. It fails for every pair, yet pairs run fine on the real scorer. Judge run time from the real durations below.

## Results

### Model comparison

| Round | Best models (masked validation) | Worst |
|---|---|---|
| 1, 1,500 sequences | `group_gru` 0.6681, `instr_gru` 0.6664, `twospeed` 0.6582, `gru_attn` 0.6581, `sru` 0.6573 | `mamba` 0.5971, `ridge` 0.5758, `esn` 0.5600 |
| 2, 5,000 sequences | `group_gru` 0.6716, `instr_gru` 0.6709, `gru2` 0.6701, `twospeed` 0.6680, `resgru` 0.6677 | `rnn` 0.6542 |

The supplied baseline scores 0.6171. `group_gru` splits the features into price, volume, trade and other groups, encodes each group separately, then feeds a GRU (a standard recurrent network). It won both rounds and is used for everything after.

### Tuning `group_gru`

- **Speed came from the export, not the model size.** Merging the per-group encoders into one matrix operation cut the ONNX operations from 83 to 24, with identical predictions.
- **Smaller models all scored lower.** Larger ones did too: `group_e32_h192`, `group_e48_h192`, `group_e32_h128_l2` and `group_e64_h256` lost 0.0007 to 0.0016 masked, and more on `cv`. The current size is the best.
- **More data helped more than more steps.** 12,000 steps added 0.0015. Adding 3,000 more sequences added 0.0042 more.
- **Averaging two models gains about 0.002** on masked validation, whether the second model has another seed, another fold or another architecture. More members gain little more: 0.0036 for 3, 0.0034 for 15.
- **Regularization:**
  - EMA of the weights at 0.999 passes the keep rule (+0.0015), mostly by rescuing a run that stopped early. EMA is a slowly updated average of the weights, and it is what gets scored and kept.
  - Dropout 0.1 hurts.
  - Input noise does nothing.
  - Weight decay 1e-4 to 1e-2 does nothing. At 0.1 it narrowly passes the keep rule locally, but it did not help a pair or the public score.
- **Training on validation half A:** masked +0.0006 on half B across 6 runs (2 seeds by 3 folds, gains on 5 of the 6), all rows +0.0037, `cv` +0.0012. This was below the bar locally, but confirmed on the public score (below).

### Public scores

| Zip | Public | Masked validation | Time |
|---|---|---|---|
| `baseline_submission.zip` (supplied) | 0.59353 | 0.6171 | 26m37s |
| `final_group_gru_24k.zip`, one model, all sequences, 24,000 steps | 0.65137 | 0.6802 | 22m56s |
| `interim_6800seq.zip`, one fold model | 0.65232 | 0.6798 | 24m7s |
| `speed_onnx2.zip`, two fold models averaged | 0.65408 | 0.6816 | 43m21s |
| `final_ema_pair.zip`, two EMA models, all sequences | 0.65440 | 0.6822 | 40m31s |
| `ema_wd_pair.zip`, as above with weight decay 0.1 | 0.65415 | 0.6820 | 42m6s |
| **`ema_wd_valid_pair.zip`, as above plus all of validation** | **0.65527** | not fair | 44m1s |

- **Public is about 0.028 below masked validation**, and the gap is steady. The baseline drops by a similar amount, so the gap comes from the test set, not from tuning against validation.
- **Gains on validation carry over to the public score at roughly 1:1.** Gaps under about 0.001 are noise.
- **Training on validation added 0.0011.** That's the gap between the two weight decay pairs, which differ in nothing else.
- "0/100" shows on every submission and is not a failure.

### Scorer run time

- A pair costs 1.84 times a single model. Nearly all scorer time is the network itself, with about 220 s of fixed overhead.
- The same network varies by up to 8% between runs: pairs have taken 40m31s to 44m1s.
- **Three `group_gru` models would take about 63 minutes, over the limit.** Larger single models are slower and also scored lower. So a pair of `group_gru` is the practical ceiling.
- Local timing does not predict the scorer well. Use the real durations above.

## Lessons for running experiments on this machine

- **Machine:** 32 GB RAM, RTX 3060 Ti with 8 GB, Windows.
- **Python:** use `.venv-train/Scripts/python.exe` (Python 3.11) with `PYTHONIOENCODING=utf-8`.
- **At most two processes at once,** whether training or scoring. A third once ran out of memory, and Claude Code stopped all the background jobs. Runs resume from `.ckpt.pkl`, but a run stopped before its first checkpoint starts over.
- **Never edit a bash queue script while it runs.** Bash reads the file as it goes, so an edit makes it skip or garble lines.
- **Stopping a queue can leave its Python children running.** Check with `Get-CimInstance Win32_Process` and stop them by process ID before restarting. Two copies writing to the same checkpoint would corrupt it.
- **Don't time anything while training is running.**
- **`mgu` is on hold** at the user's request. Don't resume it unless asked.

## Ideas not yet tried, each expected to gain at most about 0.001

- The validation pair without weight decay, since weight decay did nothing.
- Longer training, 24,000 steps, now that there is more data.
- Predicting which rows are scored and training harder on them. This was left out because it fits the visible validation mask, which the FAQ warns against.
