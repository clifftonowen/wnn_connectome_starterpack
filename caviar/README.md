# Caviar model comparison

Trains many sequence models the same way, so differences in score come from the
model and not the setup. Results are in `notebooks/02_caviar_results.ipynb` and
`reports/caviar_results.csv`.

## Setup

Use Python 3.11, the same version as the scorer. Install torch from the CUDA index first:

    py -3.11 -m venv .venv-train
    .venv-train\Scripts\activate
    pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements-train.txt

Run every command below from the repo root.

## Running it

    python -m caviar.folds                      # assign sequences to folds
    python -m caviar.data --split r1            # build the disk cache (also r2, valid)
    python -m caviar.check_models               # test every model for leakage and row-by-row match
    python -m caviar.score_baseline             # score the supplied baseline the same way
    python -m caviar.speed                      # time every model on one thread
    python -m caviar.run_all --round r1 --parallel 2
    python -m caviar.run_all --round r2 --parallel 2 --keys group_gru instr_gru --seeds 0 1 2
    python -m caviar.run_all --collect          # rebuild reports/caviar_results.csv
    python -m caviar.verify_top caviar_runs/r2/group_gru_s0_f0.pt

Stopping is safe. Finished runs are skipped, and a stopped run continues from its
`.ckpt.pkl` file. Extra arguments to `run_all`, such as `--gpu-fraction 0.3`, are
passed on to `train`.

## Files

| File | What it does |
|---|---|
| `folds.py` | Stratified folds at the sequence level, so no sequence is split across folds |
| `data.py` | Disk cache of chosen sequences, and the per-process memory cap |
| `models.py` | All models behind one interface, including row-by-row mode |
| `train.py` | Trains and scores one model on one fold |
| `gbm.py` | LightGBM on moving averages, the only non-neural model |
| `run_all.py` | Runs many jobs, and gathers results |
| `speed.py` | ONNX export and the time per row |
| `check_models.py` | Leakage and row-by-row tests on untrained models |
| `check_resume.py` | Tests that a stopped run resumes correctly |
| `score_baseline.py` | Scores the baseline, and `--run` tests a trained model |
| `verify_top.py` | Runs a model through the real scorer twice |

## Limits on this machine (32 GB RAM, 8 GB GPU)

- **Run at most 2 jobs at once.** More than that can overfill GPU memory, and everything
  becomes about 10 times slower. The sign is 100% GPU use at low power draw.
- **Large models need more GPU memory.** `retnet` and `mamba` ran out at the default
  45% cap, so they ran alone with `--gpu-fraction 0.9`.
- **`mgu` is slow.** Its gate depends on its own state, so training has to step through rows
  one at a time, at about 2.3 hours per fold.
- **`plstm` cannot export to ONNX**, because torch does not support LSTM projections.
  It has scores but no speed number.
- **Timing runs need a quiet machine.** Other jobs inflate the time per row.
