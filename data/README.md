Raw and intermediate datasets are not tracked in this repository.

The AG News dataset is downloaded automatically from Hugging Face by running:

```python
src/prepare_data.py
```

The remaining intermediate files are generated sequentially by the scripts in src/.

Expected generated data structure

```text
data/
├── processed/
│   └── ag_news_small/
├── semantic_business_scores.csv
└── cov_shift_semantic_business_exp_b12/
    ├── shifted_dataset/
    ├── control_same_n_dataset/
    ├── source_x.csv
    ├── target_x.csv
    └── target_loss_lookup.csv
```

These files are excluded from Git version control because they can be reproduced from the public AG News dataset using the scripts provided in this repository.

Model checkpoints are also not tracked. Running:

```python
src/train_semantic_pilot_frozen.py
```

creates the pilot-model checkpoint required by later steps in the pipeline.
