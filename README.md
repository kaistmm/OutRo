# On the Nature of Attention Sink that Shapes Decoding Strategy in Omni-LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2603.14337-b31b1b.svg)](https://arxiv.org/abs/2603.14337)

Official implementation of **OutRo**, accepted at **NeurIPS 2026**.  

OutRo is a training-free inference-time method for Omni-LLMs.  
This release includes AVHBench and DailyOmni inference. The default model
is Qwen2.5-Omni-3B. A single `--outro` flag enables both components of OutRo from the paper.
Without the flag, inference uses the original Qwen2.5-Omni baseline.

## Installation

Run commands from the repository root.

```bash
conda create -n outro python=3.10 -y
conda activate outro
conda install -c conda-forge ffmpeg -y

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install packaging ninja
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

## AVHBench

### Data

Download the videos with audio from [AVHBench](https://github.com/kaist-ami/AVHBench).
Set `--video-dir` to the directory containing the `.mp4` files.
The included [json/avhbench.json](json/avhbench.json) contains 5,302 Yes/No questions across
audio-driven video hallucination, video-driven audio hallucination, and AV matching.

```text
/path/to/AVHBench/videos/
  00001.mp4
  ...
```

### Inference

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_avhbench.py \
  --video-dir /path/to/AVHBench/videos
```

Full OutRo:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_avhbench.py \
  --outro \
  --video-dir /path/to/AVHBench/videos
```

Both modes use the same prompt, audio/video inputs, and greedy one-token decoding.
To use Qwen2.5-Omni-7B, add `--model-name Qwen/Qwen2.5-Omni-7B`.
Model weights are downloaded from Hugging Face.

Predictions are saved to `outputs/avhbench_3b_baseline.jsonl` and
`outputs/avhbench_3b_outro.jsonl`, with a corresponding `.summary.json` for each run.

### Evaluation

```bash
python eval/eval_avhbench.py outputs/avhbench_3b_baseline.jsonl
python eval/eval_avhbench.py outputs/avhbench_3b_outro.jsonl
```

A complete run reports `evaluated: 5302` and `missing: 0`.
The evaluator reports overall and per-task accuracy in percent, counts invalid
responses as incorrect.

## DailyOmni

### Data

Download `Videos.tar` from the official [DailyOmni dataset](https://huggingface.co/datasets/liarliar/Daily-Omni/tree/main)
and extract it under your Daily-Omni directory. See the
[official repository](https://github.com/Lliar-liar/Daily-Omni) and
[paper](https://arxiv.org/abs/2505.17862) for benchmark details.

The included [json/dailyomni.json](json/dailyomni.json) contains the 1,197-question. Set `--video-dir` to the Daily-Omni
root containing `Videos/<video_id>/<video_id>_video.mp4`.
The bundled JSON uses paths relative to the dataset root, so no path editing is needed.

```text
/path/to/Daily-Omni/
  Videos/
    <video_id>/
      <video_id>_video.mp4
```

### Inference

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_dailyomni.py \
  --video-dir /path/to/Daily-Omni
```

Full OutRo:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_dailyomni.py \
  --outro \
  --video-dir /path/to/Daily-Omni
```

Both modes use greedy one-token decoding. Predictions are saved to
`outputs/dailyomni_3b_baseline.jsonl` and `outputs/dailyomni_3b_outro.jsonl`.
Each run also writes a `.summary.json` with accuracy and valid-response counts.
video paths are relative to `--video-dir` or absolute.

### Evaluation

```bash
python eval/eval_dailyomni.py outputs/dailyomni_3b_baseline.jsonl
python eval/eval_dailyomni.py outputs/dailyomni_3b_outro.jsonl
```

A complete run reports `evaluated: 1197` and `missing: 0`.
The evaluator reports accuracy in percent and counts invalid responses as incorrect.

## Code structure

```text
inference/
  inference_avhbench.py       AVHBench loading, preprocessing, inference, and scoring
  inference_dailyomni.py      DailyOmni loading, preprocessing, inference, and scoring
model/
  modeling_qwen2_5_omni.py    Qwen2.5-Omni with OutRo integration
  outro.py                   OutRo components
  __init__.py                baseline / OutRo model loader
eval/
  eval_avhbench.py            AVHBench prediction scoring
  eval_dailyomni.py           DailyOmni prediction scoring
json/
  avhbench.json               5,302 Yes/No questions
  dailyomni.json              1,197 converted multiple-choice questions
```

## Acknowledgments

The Qwen2.5-Omni implementation is adapted from
[Transformers](https://github.com/huggingface/transformers).
We use the annotations and videos provided by [AVHBench](https://github.com/kaist-ami/AVHBench)
and [DailyOmni](https://github.com/Lliar-liar/Daily-Omni).

## Citation

```bibtex
@inproceedings{yoo2026nature,
  title={{On the Nature of Attention Sink that Shapes Decoding Strategy in Omni-LLMs}},
  author={Yoo, Suho and Jang, Youngjoon and Chung, Joon Son},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

[Apache-2.0](LICENSE). Third-party attribution is listed in [NOTICE](NOTICE).
Dataset annotations and pretrained weights retain their original terms.
