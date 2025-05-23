#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict, Counter
import re
from tqdm import tqdm

METADATA_CATEGORIES = {
    "risk_level":      ["low", "medium", "high"],
    "identifier_type": ["direct", "indirect"],
    "prompt_type":     ["Zero-Shot", "Few-Shot", "Chain-of-Thought"],
    "domain":          ["Medical", "Legal", "Government", "Common"],
    "text_length":     ["Short Sentences", "Short Paragraphs", "Long Paragraphs"],
}

DIRECT_PLACEHOLDERS = {"[ID]", "[El. paštas]", "[Tel. numeris]", "[Vardas_Pavardė]"}

PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]+\]")

def extract_placeholders(text):
    return Counter(PLACEHOLDER_PATTERN.findall(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="If set, only validate this many examples"
    )
    args = parser.parse_args()

    with open("validation.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    if args.limit:
        val_data = val_data[: args.limit]

    global_tp = global_fp = global_fn = 0
    per_example_f1 = []
    metadata_list = []

    for idx, ex in enumerate(tqdm(val_data, desc="Validating", unit="ex")):
        instr = ex["instruction"]
        inp   = ex["input"]
        gold  = ex["output"]
        meta  = ex.get("metadata", {})
        metadata_list.append(meta)

        from anonymizer import anonymize
        pred = anonymize(instr, inp, max_new_tokens=512)

        gold_ph = extract_placeholders(gold)
        pred_ph = extract_placeholders(pred)

        tp = sum(min(gold_ph[t], pred_ph.get(t, 0)) for t in gold_ph)
        fp = sum(count for t, count in pred_ph.items() if t not in gold_ph) + \
             sum(max(0, pred_ph[t] - gold_ph[t]) for t in gold_ph)
        fn = sum(count for t, count in gold_ph.items() if t not in pred_ph) + \
             sum(max(0, gold_ph[t] - pred_ph.get(t, 0)) for t in gold_ph)

        direct_missed = any((t in gold_ph and pred_ph.get(t, 0) == 0) for t in DIRECT_PLACEHOLDERS)

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp + fn == 0 else 0.0)
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        if direct_missed:
            f1 = 0.0
        else:
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        global_tp += tp
        global_fp += fp
        global_fn += fn
        per_example_f1.append(f1)

        tqdm.write(f"\n— Example {idx + 1} —")
        tqdm.write(f"Instruction: {instr}")
        tqdm.write(f"Input:       {inp}")
        tqdm.write(f"Prediction:  {pred}")
        tqdm.write(f"Reference:   {gold}")
        tqdm.write(f"TP, FP, FN:  {tp}, {fp}, {fn}")
        tqdm.write(f"Precision:   {prec:.2f}")
        tqdm.write(f"Recall:      {rec:.2f}")
        tqdm.write(f"F1:          {f1:.2f} (direct missed: {direct_missed})\n")

    micro_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    micro_rec  = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    macro_f1 = sum(per_example_f1) / len(per_example_f1) if per_example_f1 else 0.0

    breakdown = {}
    for field, choices in METADATA_CATEGORIES.items():
        counts = {}
        for choice in choices:
            total   = sum(1 for m in metadata_list if m.get(field) == choice)
            correct = sum(
                1 for m, f1_score in zip(metadata_list, per_example_f1)
                if m.get(field) == choice and f1_score == 1.0
            )
            counts[choice] = f"{correct}/{total}"
        breakdown[field] = counts

    print("\n=== Micro Metrics ===")
    print(json.dumps({
        "precision": micro_prec,
        "recall":    micro_rec,
        "f1":        micro_f1
    }, indent=2, ensure_ascii=False))

    print("\n=== Macro F1 (average per-example) ===")
    print(f"{macro_f1:.4f}")

    print("\n=== Metadata Breakdown (perfect placeholder match) ===")
    print(json.dumps(breakdown, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()