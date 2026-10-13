# Code Review and Fixes (October 2026)

A review of the whole repository: both Streamlit apps, `plant_model`/`cosinsim`, all six notebooks, the model files and the README. Each fix is its own commit, and every commit message records how the problem was reproduced and how the fix was verified.

## Findings and fixes

| # | Severity | Finding | Commit | Evidence (before → after) |
|---|----------|---------|--------|---------------------------|
| 1 | Critical | Live Weights & Biases API key hard-coded in two notebooks (`!wandb login <key>`) | `0ba94b6` | key found in 2 files → 0; **still in history, revoke it** |
| 2 | Critical | The web app could not predict at all: it loaded a missing 45-class file, while the shipped `best_model.pth` is a 38-class ResNet-50 | `7fe4a73` | `FileNotFoundError` / state-dict key mismatch → 32/33 sample images correct |
| 3 | High | Wrong preprocessing for the shipped model (128px, mean/std 0.5 instead of 224px, ImageNet) | `7fe4a73` | same weights: 8/33 → 32/33 |
| 4 | High | One visitor's OpenAI key was stored in `os.environ` and reused for every later visitor; with no key the app sent `Bearer None` | `9f48ef0` | user B's request carried `Bearer sk-USER-A` → no API call |
| 5 | High | `app.py` (the documented entry point) was a corrupted copy: `_init_`, `.replace('', ' - ')` | `fdaf7fb` | model with 0 parameters, `AttributeError` → runs `deploy.py` |
| 6 | High | 11 of 45 labels in `class_names.json` had lost underscores | `020cb43` | matches labels printed during training: False → True |
| 7 | Medium | About / How It Works pages rendered nothing (wrong indentation) | `98d60e5` | subheaders `[]` → rendered |
| 8 | Medium | Results vanished on the next rerun; a non-image upload crashed the page | `948fa12` | `[]` after rerun → kept; traceback → friendly error |
| 9 | Medium | Local LLM (Llama-2/Phi-2) never worked (`device_map` needs accelerate, `load_in_8bit` rejected); `transformers` was required even for offline mode | `04cb02d` | always fell back → generates text; app runs without transformers |
| 10 | Medium | Advice cache recreated on every rerun | `5dfbd42` | 2 API calls for one image → 1 |
| 11 | Medium | Temperature setting never reached the API | `9f48ef0` | sent 0.7 after choosing 0.1 → 0.1 |
| 12 | Medium | Offline advice never matched the 45-class labels; healthy leaves were told "This appears to be a plant disease" | `15649a3` | 0 specific entries / 14 of 14 healthy labels wrong → 18 / 0 |
| 13 | Medium | `site.py` crashed with no upload, only worked when started from its own folder, and reloaded the 91 MB model on every click | `a33f50d` | `TypeError`/`ValueError` → prompts; 1.4–1.9 s → 0.49 s per click |
| 14 | Low | In-app text claimed "50+ species" and Cotton | `01331db` | → "38 classes across 14 plant species", taken from the loaded model |
| 15 | Low | `cosinsim.similarity()` re-embedded all 64 reference images per query | `1b0f61d` | identical scores (max diff 7e-8), 2.85 s → 0.043 s per query |
| 16 | Low | Syntax errors in notebook cells (`import albumentations as `) | `3456597` | 6 cells failing → 0 |
| 17 | Low | No requirements files, tests, CI or `.gitignore`; committed `.ipynb_checkpoints` | `24606cd` `801db67` `5bfd8e8` `524c96e` | clean venv install + 53 tests (each fix covered; checked by re-adding bugs) |
| 18 | Low | README errors ("ResNet-100", 45 classes, TF-only stack, unclosed code block) | `12ca17d` | every number now cites its source notebook |

## Recommendations (need a decision, not done here)

1. **Revoke the W&B key** at https://wandb.ai/authorize. It is still in commit `69039e3`. Removing it from history means rewriting history (`git filter-repo`) and force-pushing, which changes every later commit hash.
2. **Re-measure accuracy without leakage.** The dataset contains augmented copies of each leaf (`_270deg`, `_flipTB`, …), and the custom CNN used a `random_split` without a seed, so copies of the same leaf can be in both train and test. Split by original leaf ID (strip the augmentation suffix) with a fixed seed, and test on field photos.
3. **Move large binaries to Git LFS.** `best_model.pth` is 98.6 MB, just under GitHub's 100 MB per-file limit.
4. **Publish the 45-class checkpoint** (`plant_disease_modelFinal_45Class15.pth`). The app already supports it; without it, rice and wheat can't be detected.
5. **Remove duplicates:** `cv-projectnewupdated (8)`/`(9)` notebooks, and two same-size report PDFs.
6. **Rejecting non-leaf images:** `cosinsim.similarity()` could flag non-leaf uploads once a threshold is calibrated on real examples.
