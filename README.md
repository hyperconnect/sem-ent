# Measuring and Improving Semantic Diversity of Dialogue Generation (ENMLP 2022 Findings)

## Paper link

- https://arxiv.org/abs/2210.05725

## Overview

- `codes/evalutate_diversify.py` - Measuring semantic diversity of dialogue generation.
- `codes/train_dress.py`, `train_diversify_base.py` - Training the model to improve the semantic diversity of dialogue generation.
- `codes/dm_generator.py` - A generator agent for DRESS.

## Installation

- First, this code is implemented based on [ParlAI](https://github.com/facebookresearch/ParlAI) and [Huggingface Transformers](https://github.com/huggingface/transformers).
  You need to install ParlAI and Huggingface Transformers as described in the README on those links.

- After installing ParlAI on your local, then move the codes as follows:
  - `codes/evalutate_diversify.py` -> `parlai/scripts/evalutate_diversify.py`
  - `codes/train_dress.py` -> `parlai/scripts/train_dress.py`
  - `codes/train_diversify_base.py` -> `parlai/scripts/train_diversify_base.py`
  - `codes/dm_generator.py` -> `parlai/agents/transformer/dm_generator.py`
  - `codes/utils/` -> `parlai/utils`

- Install mauve package in `./mauve`:
  ```bash
  pip install -e mauve
  ```

## Training & Evaluation

### Training

```bash
# Blender 90M
bash scripts/train_balancing.sh [init_model] 32 7e-6 dailydialog

# Bart-large
bash scripts/train_balancing_bart.sh [init_model] 16 7e-6 dailydialog
```

### Evaluation

```bash
bash scripts/eval_dailydialog.sh [model_filename] [report_filename] 32
```

## Citation

Please cite the following if you make use of this codebase.

```
@article{han2022measuring,
  title={Measuring and Improving Semantic Diversity of Dialogue Generation},
  author={Han, Seungju and Kim, Beomsu and Chang, Buru},
  journal={arXiv preprint arXiv:2210.05725},
  year={2022}
}
```
