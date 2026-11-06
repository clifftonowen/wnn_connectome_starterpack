# Caviar tuning: overnight plan

Goal: a model well under the speed limit that keeps as much score as possible.
Approved by the user for an unattended overnight run, stopping only on failure.

Fixed comparison: Round 2 cache, folds 0, 1 and 2, seed 0. These folds and seed match
existing runs, so comparisons are paired. Picks use masked validation, with the
held-out fold score reported beside it.

Speed target: about 75 µs per row in the real scorer, and 78 µs or less counts as safe.
Timing runs only when no training or cache build is running.

## Steps

1. **`gru2` Round 2.** 5 folds, seed 0. It "holds up" if its masked mean is at least
   0.6659, which is within 0.005 of `instr_gru` (0.6709). If it does not hold up, its
   small versions are dropped from step 2.
2. **Where the time goes.** Measure the scorer's own cost with a constant prediction,
   and the ONNX call's cost with a tiny model. The rest is what shrinking can save.
3. **Size sweep.**
   - Add small versions of `instr_gru` and `gru2`, kept apart from the original 30 models.
     Also try fewer ONNX operations, not only narrower layers.
   - Test them for leakage and row-by-row match.
   - Time each through the real scorer on a subset of validation, a few minutes each.
   - Train up to 6, nearest the speed target, on the fixed comparison.
   - Pick the best masked mean at 78 µs or less. Within 0.001 masked, take the faster.
   - Two full scorer passes on the pick. If it measures above 78 µs, check the next pick,
     up to 2 models.
   - Fallback if nothing reaches 78 µs: the fastest model within 0.003 masked of
     `instr_gru`. Carry on with it and flag it in the write-up.
4. **Longer training and more sequences.** On the model chosen in step 3:
   - **A:** 12,000 steps instead of 3,000.
   - **B:** 12,000 steps plus 3,000 extra sequences used only for training. They come from
     sequences outside Round 2, so held-out folds stay clean. This adds about 27 GB of
     cache and leaves about 35 GB free.
   - Both use a dev check every 500 steps with patience 5, because more checks on a long
     run add noise and risk stopping too early.
   - A against B is decided by paired masked mean over the 3 folds.
   - One full scorer pass on the better of the two, if it beats the step 3 model.
5. **Write-up.** Tuning notebook with results, what to do next, and any failures. State
   whether longer training helped from the step 4 result. The cosine schedule ends at the
   step cap, so reaching the cap alone never showed that more steps help. Correct that
   claim in `notebooks/02_caviar_results.ipynb`.

Commit and push after each step, on `caviar-model-comparison`, using explicit paths so the
EDA files stay out.

## If time runs short

Drop step 4B first, then reduce the sweep candidates.

## Out of scope

Resuming `mgu`, packaging a submission, and merging to `main`.
