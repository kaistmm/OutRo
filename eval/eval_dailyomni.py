import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score(paths, data):
    with open(data, encoding='utf-8') as handle:
        annotations = json.load(handle)
    gold = {
        item.get('id', index): item['conversations'][1]['value'].strip()[0].upper()
        for index, item in enumerate(annotations)
    }
    seen = set()
    correct = invalid = 0
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                key = row['id']
                item = gold[key]
                seen.add(key)
                response = row['response'].strip().upper()
                pred = next((c for c in 'ABCD' if response.startswith(c) or f'({c})' in response or f'[{c}]' in response), None)
                if pred is None:
                    match = re.search(r'Answer:\s*([A-D])', response, re.IGNORECASE)
                    pred = match.group(1).upper() if match else None
                ok = pred == item
                correct += int(ok)
                invalid += int(pred is None)
    return dict(
        accuracy=100 * correct / len(seen) if seen else 0.0, correct=correct, evaluated=len(seen),
        expected=len(gold), missing=len(gold) - len(seen), invalid=invalid,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', nargs='+', help='One result file or disjoint shards of one run.')
    parser.add_argument('--data', type=Path, default=ROOT / 'json/dailyomni.json')
    args = parser.parse_args()
    print(json.dumps(score(args.predictions, args.data), indent=2))
