import argparse
import json
import os
import re


def parse_accuracy_from_text(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if not m:
        return None
    return float(m.group(1))


def parse_accuracy(stage_dir: str):
    science_path = os.path.join(stage_dir, "output_result.jsonl")
    if os.path.isfile(science_path):
        with open(science_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "acc" in obj:
            return float(obj["acc"])

    result_text_path = os.path.join(stage_dir, "Result.text")
    if os.path.isfile(result_text_path):
        acc = parse_accuracy_from_text(result_text_path)
        if acc is not None:
            return acc

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    args = parser.parse_args()

    acc = parse_accuracy(args.stage_dir)
    if acc is None:
        raise SystemExit("Could not parse accuracy from stage dir: {}".format(args.stage_dir))
    print("{:.6f}".format(acc))


if __name__ == "__main__":
    main()
