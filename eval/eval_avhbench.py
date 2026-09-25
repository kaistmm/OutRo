import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score(paths, data):
    with open(data, encoding='utf-8') as handle:
        annotations = json.load(handle)
    gold = {
        index: {**item, 'answer': str(item['label']).strip().capitalize()}
        for index, item in enumerate(annotations)
        if str(item.get('label', '')).strip().lower() in {'yes', 'no'}
    }
    seen = set()
    correct = invalid = 0
    by_task = {}
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                key = row['id']
                item = gold[key]
                seen.add(key)
                match = re.search(r'\b(yes|no)\b', row['response'].strip(), re.IGNORECASE)
                pred = match.group(1).capitalize() if match else None
                ok = pred == item['answer']
                correct += int(ok)
                invalid += int(pred is None)
                stats = by_task.setdefault(item['task'], {'correct': 0, 'evaluated': 0})
                stats['correct'] += int(ok)
                stats['evaluated'] += 1
    for stats in by_task.values():
        stats['accuracy'] = 100 * stats['correct'] / stats['evaluated']
    return dict(
        accuracy=100 * correct / len(seen) if seen else 0.0, correct=correct, evaluated=len(seen),
        expected=len(gold), missing=len(gold) - len(seen), invalid=invalid, by_task=by_task,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', nargs='+', help='One result file or disjoint shards of one run.')
    parser.add_argument('--data', type=Path, default=ROOT / 'json/avhbench.json')
    args = parser.parse_args()
    print(json.dumps(score(args.predictions, args.data), indent=2))
