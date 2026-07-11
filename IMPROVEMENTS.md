# IMPROVEMENTS.md

*Analysis date: 2026-07-11*

This repository is a feed-forward neural network baseline for the ultratree /
ultrametric-trees experiments: `ffnn-senses.py` trains a simple
embedding + 2-layer FFNN next-word-sense predictor from a prepared SQLite
dataframe, `evaluate_model.py` scores it on held-out data using an
ultrametric path penalty (`compute_penalty`), and `summarise_evaluations.py`
plus the Makefile sweep 16 model sizes (sense-annotated and unannotated,
emb 2..128) into `../ultratree-results/neural-results.csv`. The code works
end-to-end (there are committed `.out` results for every configuration), but
it carries research-prototype debt: duplicated model code, AI-generated tests
the README itself says "probably don't work", committed binary/backup
artifacts, and no packaging.

## Bugs & Fixes

- **Duplicated `SimpleFFNN` definition.** The class is defined independently
  in `ffnn-senses.py` and `evaluate_model.py`. Any architecture change (e.g.
  the "noun loss" work in commit 009088a) must be made twice or evaluation
  silently loads mismatched weights. Extract a shared `model.py` module and
  import it from both scripts. (The filename `ffnn-senses.py` has a hyphen so
  it can't be imported — another reason to move the class out, or rename to
  `ffnn_senses.py`.)
- **`--context-size` can silently disagree with the data schema.**
  `WordSenseDataset.load_data()` hard-codes `context1..context16` and
  `build_word_sense_vocab` uses the module-level `CONTEXT_SIZE = 16`, but
  `main()` accepts `--context-size` and builds the model with it. Passing
  anything other than 16 produces a shape mismatch at the first batch (or a
  model that never sees some columns). Either derive context size from the
  table columns or validate `args.context_size == 16`.
- **`torch.load(..., weights_only=False)`** in `evaluate_model.py`
  (`load_model`) executes arbitrary pickles. Fine for your own checkpoints,
  but consider saving the vocab as JSON alongside the state dict so you can
  use `weights_only=True`.
- **OOV embedding is trained as a real token.** `padding_idx=0` zeroes the
  OOV embedding's gradient, yet OOV also appears as a *target* class in
  `CrossEntropyLoss`, so the model can predict OOV but never learn a useful
  representation for it as context. Decide deliberately: either drop
  `padding_idx` or mask OOV targets.
- **No reproducibility seed.** `random_split` and DataLoader shuffling are
  unseeded, so train/val splits differ per run; add `torch.manual_seed` and a
  `--seed` flag so sweep results are comparable.

## Improvements

- **Device support.** Training runs CPU-only; a single
  `device = "cuda" if ... else "mps" if ... else "cpu"` plus `.to(device)`
  on model and batches would cut the sweep time dramatically on this Mac
  (MPS) or on raksasa.
- **Dataset loads everything into RAM as Python tuples** in
  `WordSenseDataset.load_data()`. Convert once to a single `torch.LongTensor`
  at load time — `__getitem__` then becomes a slice, removing per-sample
  tensor construction (a large fraction of CPU training time).
- **Makefile duplication.** The 16 near-identical rules and the giant
  dependency lists can collapse to a pattern rule plus a
  `DIMS = 2_4 4_8 ...` list with `$(foreach ...)`. Right now adding a new
  size means editing four places.
- **`sweep.yaml`** suggests Sweep-bot automation was used; several merged
  `sweep/*` branches only added type annotations. If Sweep is no longer in
  use, delete `sweep.yaml`.

## Testing

- The README explicitly says `test_evaluate_model.py` and
  `test_ffnn_senses.py` "are AI-generated and probably don't work". Either
  fix them or delete them — broken tests are worse than none. The one thing
  genuinely worth a unit test is `compute_penalty` in `evaluate_model.py`
  (exact match, shared-prefix, and no-common-prefix cases), which is pure and
  trivial to test.
- Add a tiny fixture SQLite database (a few dozen rows) so a smoke test can
  run train → evaluate → summarise in seconds under CI.

## Documentation

- README's `--output-db` description says "where to write it do" (typo) and
  doesn't mention the `.out` files, `make all` prerequisites (the
  `../ultrametric-trees/*.sqlite` paths), or the emb128_hidden512 addition.
- Document the checkpoint format (keys saved in `torch.save`) since
  `evaluate_model.py` depends on it.
- Note what "noun loss" (commit 009088a) changed and whether old `.out`
  files predate it — otherwise the committed results are not comparable.

## Housekeeping / Modernization

- **Migrate to uv.** Replace `requirements.txt` (currently just `torch`,
  `numpy`) with a `pyproject.toml` via `uv init && uv add torch numpy`,
  commit `uv.lock`, and run scripts as `uv run ffnn-senses.py ...`. Update
  the Makefile's `python3` invocations to `uv run`.
- **Remove committed cruft:** `ffnn-senses.py~` (editor backup) is checked
  in; delete it. `model.pt` and `sense_annotated_model_emb4_hidden8.pt` are
  binary checkpoints in git — move large artifacts out of the repo (or to
  git-lfs) and add `*.pt`, `*~`, `*.sqlite` to a `.gitignore` (the repo has
  none).
- **Untracked `evaluation.sqlite`** is sitting dirty in `git status`; ignore
  it or archive it — it should not end up committed by accident.
- No secrets or credentials were found in the repo.

## Quick Wins

1. Add `.gitignore` (`*.pt`, `*~`, `*.sqlite`, `__pycache__/`).
2. `git rm ffnn-senses.py~`.
3. Fix the README "write it do" typo and list `make all` prerequisites.
4. Seed randomness (`--seed`, default 0).
5. `uv init` + `uv add torch numpy`, delete `requirements.txt`.
