# Results

| Path | Status | Contents |
| --- | --- | --- |
| `paper/` | **current** (the paper) | 10 m inference, bilinear to the 3 m reference grid, threshold 0.30, common valid mask: Table I, Table II, WorldCover, Fig. 1B panel values, Supplementary Table S2 |
| `paper_labelclass_v1/58321212/` | superseded (3 m-input protocol) | label-class models evaluated with inputs resampled to the 3 m reference grid; each method on its own valid pixels in Table I |
| `paper_retrain_openwater_v1/57725872/` | superseded (3 m-input protocol, mixed labels) | runs trained on a mixture of Dynamic World `label` and binarized water probability |
| `paper_metrics.csv`, `ablation_metrics.csv` | superseded (first release, 20 August 2026) | mixed-label models trained on 3,742 tiles, before the open-water supplement |
| `released_model_training.json` | superseded (first release, 20 August 2026) | training record of the `v1-mixed-labels` checkpoint (run `train_53123937`); the `v2-labelclass` checkpoints are listed in `models/model_registry.json` |

The superseded files are kept unchanged as provenance for earlier manuscript versions. Numbers in them (for
example S1-only pooled IoU 0.558, per-scene 0.443 [0.355, 0.533], previous-year AEF 0.840) are not the paper's
results.
