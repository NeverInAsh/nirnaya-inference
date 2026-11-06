# Nirnaya inference

Inference code for the Nirnaya decision family, for use with the [Decision Index kit](https://github.com/apolinario/decision-index). It covers running the models only.

## Models

| Model | Weights |
|---|---|
| Nirnaya Flash 1.0 | `Darkrider/nirnaya-flash-1.0` |
| Nirnaya Turbo 1.0 | `Darkrider/nirnaya-turbo-1.0` |

## Setup

```bash
pip install vllm==0.30.0 transformers==5.17.0 torch==2.13.0
pip install -e "git+https://github.com/apolinario/decision-index@25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d#egg=decision_index"
export PYTHONPATH=$PWD
```

## Text suite

```bash
python -m decision_index run --engine mmclm.di_engine:NirnayaEngine \
  --option model=<model dir> --option head=<model dir>/head.pt \
  --option temperature_fit=<model dir>/t_entropy.json \
  --option twice_max=3000 --option max_model_len=32768 --option gpu_mem=0.85 \
  --edition 0.3 --rows <rows.jsonl> --out <run dir> --compact
```

## Vision suite

Use `mmclm.di_engine:NirnayaVisionEngine` through `decision-index vision run`, with the options above plus `max_soft_tokens=560` and `max_images=2`.

| File | What it does |
|---|---|
| `mmclm/di_engine.py` | The two engines |
| `mmclm/joint.py` | Prompt layout |
| `mmclm/jh.py` | The decision head and how it reads the model's hidden states |
