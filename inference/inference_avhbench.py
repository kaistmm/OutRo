import argparse
import json
import os
import re
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from qwen_omni_utils import process_mm_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import load_pretrained_model_flash


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


class AvhDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        if not re.search(r"\banswer\s+yes\s+or\s+no\b", question, re.IGNORECASE):
            question += " Answer yes or no."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "video", "video": sample["video_path"]},
                {"type": "text", "text": question},
            ]},
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        audios, _, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(
            text=text, audio=audios, videos=videos, return_tensors="pt",
            padding=True, use_audio_in_video=True,
        )
        return inputs, {**sample, "question": question}


def collate_fn(batch):
    return batch[0]


def parse_yes_no(response):
    match = re.search(r"\b(yes|no)\b", response.strip(), re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def load_samples(question_file, video_dir):
    """Read the official QA.json or an already filtered Yes/No question file."""
    with open(question_file, encoding="utf-8") as handle:
        raw = json.load(handle)
    samples = []
    for index, sample in enumerate(raw):
        label = str(sample.get("label", "")).strip().lower()
        if label not in {"yes", "no"}:
            continue  # AV Captioning is not part of the Yes/No evaluation.
        video_id = str(sample["video_id"])
        video_name = video_id if video_id.endswith(".mp4") else video_id + ".mp4"
        samples.append({
            "id": index,
            "video_id": video_id,
            "video_path": str(Path(video_dir) / video_name),
            "task": sample.get("task", "unknown"),
            "question": sample["text"].strip(),
            "answer": label.capitalize(),
        })
    return samples


def resolve_output_file(args):
    size = args.model_name.rsplit("-", 1)[-1].lower()
    mode = "outro" if args.outro else "baseline"
    return Path(args.output_dir) / f"avhbench_{size}_{mode}.jsonl"


def run_inference(args):
    samples = load_samples(args.question_file, args.video_dir)
    output_file = resolve_output_file(args)
    model, processor = load_pretrained_model_flash(
        model_name=args.model_name, use_outro=args.outro, cache_dir=args.cache_dir,
    )
    dataloader = DataLoader(
        AvhDataset(samples, processor), batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    valid = 0
    by_task = {}
    print(f"Model: {args.model_name}; mode: {'full OutRo' if args.outro else 'baseline'}")
    print(f"Questions: {len(samples)}; output: {output_file}")

    with output_file.open("w", encoding="utf-8") as handle:
        for inputs, sample in tqdm(dataloader):
            inputs = inputs.to(model.device).to(model.dtype)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                text_ids = model.generate(
                    **inputs, use_audio_in_video=True, return_audio=False,
                    thinker_max_new_tokens=1, thinker_do_sample=False,
                )
            response = processor.batch_decode(
                text_ids[:, prompt_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            pred = parse_yes_no(response)
            is_correct = pred == sample["answer"]
            correct += int(is_correct)
            valid += int(pred is not None)
            task_stats = by_task.setdefault(sample["task"], {"correct": 0, "total": 0})
            task_stats["correct"] += int(is_correct)
            task_stats["total"] += 1
            handle.write(json.dumps({
                **sample, "response": response, "pred": pred, "is_correct": is_correct,
            }, ensure_ascii=False) + "\n")
            handle.flush()

    total = len(samples)
    accuracy = correct / total if total else 0.0
    for stats in by_task.values():
        stats["accuracy"] = stats["correct"] / stats["total"]
    summary = {
        "model": args.model_name, "outro": args.outro,
        "question_file": str(args.question_file), "video_dir": str(args.video_dir),
        "decoding": "greedy", "max_new_tokens": 1,
        "correct": correct, "total": total, "accuracy": accuracy,
        "valid_responses": valid, "by_task": by_task,
    }
    output_file.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total}); valid: {valid}/{total}")
    for task, stats in by_task.items():
        print(f"  {task}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outro", action="store_true",
                        help="Enable full OutRo: ReLU-tanh gated head output rotation and "
                             "sink information enhancement via mask relaxation.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Omni-3B",
                        choices=["Qwen/Qwen2.5-Omni-3B", "Qwen/Qwen2.5-Omni-7B"])
    parser.add_argument("--question-file", default=os.environ.get("QUESTION_FILE", str(ROOT / "json/avhbench.json")))
    parser.add_argument("--video-dir", default=os.environ.get("VIDEO_DIR", "data/AVHBench/videos"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cache-dir", default=os.environ.get("MODEL_CACHE_DIR"))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_inference(parse_args())
