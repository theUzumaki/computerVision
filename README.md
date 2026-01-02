# Computer Vision Project Template

Organized into four independent layers for parallel development:

- **data/** — loading and splitting datasets
- **cv_enhancement/** — image preprocessing, augmentation, filters
- **nn_training/** — model creation and training
- **validation/** — metrics, evaluation, visualization

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure paths in `configs/default_config.yaml`.

```python
from configs.loader import load_config, Config
cfg = Config(load_config())
```

---

## Layers

### Data (`data/`)

Implement `DataLoader` interface in [data/dataloader.py](data/dataloader.py):

- `load(path: str)` — load dataset from path
- `split(data)` — return `(train, val, test)` tuple

### CV Enhancement (`cv_enhancement/`)

Implement `CVEnhancementAPI` in [cv_enhancement/api/interface.py](cv_enhancement/api/interface.py):

- `preprocess(dataset, **kwargs)` — dataset-level preprocessing; accepts a `DataFrame` or `list[dict]` with `file_path` and **adds** an `image` field containing processed arrays (no resizing is performed)
- `augment(image, **kwargs)` — apply data augmentation
- `apply_filter(image, filter_name, **kwargs)` — apply image filters

### Training (`nn_training/`)

Implement `Trainer` class:

- `create_model(model_cfg: dict)` — create model from config
- `train(model, train_data, val_data, **kwargs)` — train and return history

Required kwargs: `epochs`, `batch_size`, `learning_rate`, `checkpoint_dir`, `save_frequency`, `device`, `resume`, `seed`

History format:
```python
{
  "train_loss": [...],      # per epoch
  "val_loss": [...],        # per epoch
  "train_acc": [...],       # optional
  "val_acc": [...],         # optional
  "best_checkpoint": "path/to/best.pth",
  "best_metric": 0.87
}
```

Run:
```bash
python -m nn_training.train --trainer pkg.Trainer --dataloader pkg.DataLoader --exp-name myrun
```

### Validation (`validation/`)

Implement `ValidationAPI` in [validation/api/interface.py](validation/api/interface.py):

- `compute_metrics(predictions, targets, metrics: list)` — compute metrics dict
- `evaluate(model, test_data, **kwargs)` — evaluate and return results dict
- `plot_results(results, plot_type, save_path)` — generate visualizations
- `plot_from_history(history, plot_types, save_path)` — plot training curves

---

## Pipeline

Run full pipeline:
```bash
python scripts/run_pipeline.py --exp-name quicktest
```

Each run creates `experiments/<run-id>/` with checkpoints, logs, results, and metadata.

---

## Development

- Implement the interface for your layer
- Use config values via `Config` helpers
- Keep history format consistent for comparison
- Each experiment is self-contained and reproducible
