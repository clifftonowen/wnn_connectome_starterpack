# Runs after the first public scores

## Public scores so far

| Zip | Public | Masked validation | Drop | Scorer time |
|---|---|---|---|---|
| baseline_submission.zip | 0.59353 | 0.6171 | 0.0236 | 26m37s |
| final_group_gru_24k.zip | 0.65137 | 0.6802 | 0.0288 | 22m56s |
| interim_6800seq.zip | 0.65232 | 0.6798 | 0.0275 | 24m7s |
| speed_onnx2.zip (two group_gru averaged) | 0.65408 | 0.6816 | 0.0275 | 43m21s |
| final_ema_pair.zip (two EMA group_gru, all sequences) | 0.65440 | 0.6822 | 0.0278 | 40m31s |
| ema_wd_pair.zip (as above, weight decay 0.1) | 0.65415 | 0.6820 | 0.0279 | 42m6s |
| ema_wd_valid_pair.zip (as above, plus all of validation) | 0.65527 | not fair | | 44m1s |

A pair took 1.84 times as long as one model, so nearly all scorer time is the network itself.
Three group_gru models would take about 63 minutes, over the 60 minute limit.

## Larger models and regularization

All runs: group_gru setup, folds 0 to 2, seed 0, 12,000 steps at most, extra sequences.
Each is compared with `r2_long_extra` group_gru on the same fold (masked 0.6777, cv 0.4471, all rows 0.5068).
Keep rule: masked and cv both rise on every fold, and the mean masked gain is over 0.001.

| Run | Masked | Change | cv | Change | All rows | Change | Per fold, masked / cv |
|---|---|---|---|---|---|---|---|
| group_e32_h192 | 0.6770 | -0.0007 | 0.4444 | -0.0027 | 0.5053 | -0.0016 | -0.0004/-0.0033, -0.0002/+0.0013, -0.0016/-0.0061 |
| group_e48_h192 | 0.6761 | -0.0016 | 0.4437 | -0.0034 | 0.5020 | -0.0048 | -0.0026/-0.0070, -0.0014/+0.0014, -0.0009/-0.0047 |
| group_e32_h128_l2 | 0.6766 | -0.0012 | 0.4435 | -0.0036 | 0.5043 | -0.0025 | -0.0020/-0.0068, -0.0019/-0.0043, +0.0005/+0.0004 |
| group_e64_h256 | 0.6764 | -0.0013 | 0.4432 | -0.0039 | 0.5034 | -0.0034 | -0.0030/-0.0069, -0.0044/-0.0075, +0.0034/+0.0026 |
| EMA 0.999 | 0.6793 | +0.0015 | 0.4491 | +0.0020 | 0.5085 | +0.0017 | +0.0001/+0.0004, +0.0004/+0.0014, +0.0042/+0.0041 |
| dropout 0.1 | 0.6766 | -0.0011 | 0.4457 | -0.0014 | 0.5051 | -0.0017 | -0.0007/-0.0011, -0.0027/-0.0028, +0.0000/-0.0004 |
| weight decay 1e-3 | 0.6778 | +0.0000 | 0.4471 | -0.0000 | 0.5068 | +0.0000 | no change |
| weight decay 1e-4 | 0.6777 | -0.0000 | 0.4471 | -0.0000 | 0.5068 | -0.0000 | no change |
| input noise 0.05 | 0.6777 | -0.0000 | 0.4469 | -0.0002 | 0.5066 | -0.0002 | -0.0001/-0.0004, +0.0000/+0.0003, -0.0000/-0.0006 |

Findings:

- **Larger models are worse, not better.** All four lose on masked and cv. More capacity overfits at this amount of data.
- **EMA passes the keep rule, mostly through fold 2.** On folds 0 and 1 it gains only 0.0001 to 0.0014. On fold 2 it lifts the weakest reference run (0.6748, stopped early at step 8,500) to 0.6789. So it mainly protects against an unlucky run.
- **Weight decay at 1e-3 and 1e-4 did nothing.** The setting was applied, but with AdamW at lr 1e-3 it shrinks the weights by under 1% over a run. A real test needs values near 0.01 to 0.1.
- **Dropout hurts, input noise does nothing.**

## Training on validation (step 4)

Validation is split into fixed halves in file order (sorted `seq_ix`, which looks random): A is the first 936 sequences, B the other 937.
The new runs add half A to training (`--valid-train A`), with EMA 0.999, on folds 0 to 2.
They are compared with `r2_ema` on half B. Each reference model first reproduced its stored full validation score; the half-B numbers are in `caviar_half_b_reference.json`.

| Fold | Half B masked, with A | Without A | Change | cv change | Half B all rows change |
|---|---|---|---|---|---|
| 0 | 0.6782 | 0.6801 | -0.0019 | -0.0047 | -0.0005 |
| 1 | 0.6811 | 0.6800 | +0.0011 | +0.0047 | +0.0050 |
| 2 | 0.6805 | 0.6787 | +0.0018 | +0.0002 | +0.0045 |
| Mean | | | +0.0003 | +0.0000 | +0.0030 |

- **It fails the keep rule, so validation stays out of training for now.**
- Fold 0 stopped early, at step 7,000 with its best at 4,500. Folds 1 and 2 ran to 12,000 and gained.
- So the result is not proven either way. A second seed on all three folds would settle it, at about 1.5 hours.

## Next candidate: final_ema_pair.zip (step 5)

Two group_gru models with EMA 0.999, trained on all training sequences (fold -1, extra and rest splits), seeds 0 and 1, averaged in one ONNX file.

- Single models on masked validation: 0.6799 and 0.6811. The pair scores **0.6822**, against 0.6816 for speed_onnx2.zip.
- Two full local passes gave identical predictions. Windows took 112 to 118 us per row.
- Linux container with one CPU, on the subset: 104 and 116 us per row, identical passes. speed_onnx2.zip took 137 and 122 in the same run, so the new pair costs about the same and should take about 43 minutes on the scorer.
- The local under-limit check fails for both, as it did for speed_onnx2, because that limit uses this machine's timing.

## Second seed of the validation test, and stronger weight decay

Seed 1 repeats the validation test against its own seed 1 EMA reference (`caviar_half_b_reference_s1.json`). Changes on half B, with half A added:

| Seed, fold | Masked | All rows | cv |
|---|---|---|---|
| s0 f0 | -0.0019 | -0.0005 | -0.0047 |
| s0 f1 | +0.0011 | +0.0050 | +0.0047 |
| s0 f2 | +0.0018 | +0.0045 | +0.0002 |
| s1 f0 | +0.0007 | +0.0052 | +0.0037 |
| s1 f1 | +0.0008 | +0.0040 | -0.0003 |
| s1 f2 | +0.0010 | +0.0037 | +0.0034 |
| Mean of 6 | +0.0006 | +0.0037 | +0.0012 |

- Masked gains on 5 of 6 runs, but the mean of +0.0006 is under the 0.001 bar. All-row gains are larger and steady.
- The only loss is the run that stopped early. Only half of validation was added here, so adding all of it could give more.
- A model trained on all of validation can't be checked on validation, so only the public score can judge it.

Weight decay with EMA, compared with `r2_ema` seed 0 (masked / cv per fold):

| Weight decay | Fold 0 | Fold 1 | Fold 2 | Mean masked | Mean cv | Mean all rows |
|---|---|---|---|---|---|---|
| 0.01 | +0.0001/+0.0003 | +0.0003/+0.0007 | -0.0003/-0.0002 | +0.0001 | +0.0002 | +0.0001 |
| 0.1 | +0.0001/+0.0001 | +0.0013/+0.0021 | +0.0017/+0.0010 | +0.0010 | +0.0011 | +0.0012 |

- 0.1 passes the keep rule narrowly: all folds rise on both measures and the mean masked gain is 0.00104. Fold 0 barely moves.
- 0.01 does nothing.

## Weight decay 0.1 pairs, with and without validation

Both pairs: two group_gru with EMA 0.999 and weight decay 0.1, all training sequences, seeds 0 and 1.

| Zip | Single models, masked | Pair, masked | Notes |
|---|---|---|---|
| final_ema_pair.zip (weight decay 1e-5, for comparison) | 0.6799, 0.6811 | 0.6822 | public 0.65440 |
| ema_wd_pair.zip | 0.6808, 0.6811 | 0.6820 | no gain over final_ema_pair |
| ema_wd_valid_pair.zip, plus all of validation | 0.6831, 0.6828 (trained on, not fair) | not fair | only the public score can judge it |

- Weight decay 0.1 lifted the weaker single model, but the pair did not gain.
- The models trained on validation score only about 0.002 higher on it, so they don't simply memorise it.
- Both zips gave identical predictions on two passes, on Windows and on Linux. In one Linux run, all three pairs took 88.6 to 91.0 us per row, so all should take about 40 to 43 minutes on the scorer.

## Public scores of the weight decay pairs

- ema_wd_valid_pair.zip scored 0.65527, the best so far: +0.0011 over ema_wd_pair.zip (0.65415), whose only difference is that it did not train on validation.
- That matches the all-rows gain seen on half B. Training on validation helps.
- Weight decay 0.1 itself did nothing on the public score either: 0.65415 against 0.65440.
- Scorer times for the three pairs range from 40m31s to 44m1s for the same network, so expect about 8% variation between runs.
