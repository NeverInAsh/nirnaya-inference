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

## Audio

`mmclm.di_engine:NirnayaAudioEngine` takes one or more clips with any request. `mmclm/serve.py` serves it over HTTP:

```bash
pip install librosa fastapi uvicorn python-multipart
NIRNAYA_API_KEY=<key> python -m mmclm.serve --model <model dir> --port 8000
```

`POST /v1/decisions` takes `multipart/form-data` with a `request` part (JSON with `model`, `state` and `questions`) and an `audio` part (WAV), or `application/json` for text only. It returns `{"model": ..., "answers": {<question id>: {"type": "choice", "choice": <option id>, "probabilities": {...}}}}`. It binds to 127.0.0.1 by default and handles one request at a time. The Decision Index has no audio suite, so this engine isn't part of the board submission.

| File | What it does |
|---|---|
| `mmclm/di_engine.py` | The text, vision and audio engines |
| `mmclm/serve.py` | HTTP endpoint for the audio engine |
| `mmclm/joint.py` | Prompt layout |
| `mmclm/jh.py` | The decision head and how it reads the model's hidden states |
