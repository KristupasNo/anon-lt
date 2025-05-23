#!/usr/bin/env python3
import json, re
from collections import Counter
from pathlib import Path

# --- CONFIG ---
GOLD_FILE    = "gold.jsonl"
PRED_FILES   = {
    "chatgpt":   "chatgpt.jsonl",
    "deepseek":  "deepseek.jsonl",
    "gemini":    "gemini.jsonl",
    "lt_llama":  "lt_llama.jsonl",  
}
ALPHA = 0.8

DIRECT_PLACEHOLDERS = {"[ID]", "[El. paštas]", "[Tel. numeris]", "[Vardas_Pavardė]"}
PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")

def extract_placeholders(text):
    return Counter(PLACEHOLDER_RE.findall(text))

def load_jsonl(path):
    """
    Load either a JSON array or newline-delimited JSON, stripping trailing commas.
    Skips blank or invalid lines.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass 
    records = []
    for line in text.splitlines():
        raw = line.strip().rstrip(",")
        if not raw:
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line in {path}: {raw[:80]}…")
    return records

def evaluate_model(gold, preds):
    pred_map = {r["id"]: r["pred"] for r in preds}
    # global sums
    all_tp=all_fp=all_fn=0
    dir_tp=dir_fp=dir_fn=0
    indir_tp=indir_fp=indir_fn=0
    per_f1=[]; per_f1_dir=[]

    for ex in gold:
        gid = ex["id"]
        gtxt, ptxt = ex["gold"], pred_map.get(gid, "")
        id_type = ex.get("id_type","direct")
        gph, pph = extract_placeholders(gtxt), extract_placeholders(ptxt)

        tp = sum(min(gph[t], pph.get(t,0)) for t in gph)
        fp = sum(pph[t] for t in pph if t not in gph) + \
             sum(max(0, pph[t]-gph[t]) for t in gph)
        fn = sum(gph[t] for t in gph if t not in pph) + \
             sum(max(0, gph[t]-pph.get(t,0)) for t in gph)

        missed_direct = any(gph[t]>0 and pph.get(t,0)==0 for t in DIRECT_PLACEHOLDERS)
        prec = tp/(tp+fp) if tp+fp else (1.0 if tp+fn==0 else 0.0)
        rec  = tp/(tp+fn) if tp+fn else 1.0
        if id_type=="direct" and missed_direct:
            f1 = 0.0
        else:
            f1 = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

        all_tp+=tp; all_fp+=fp; all_fn+=fn
        per_f1.append(f1)
        if id_type=="direct":
            dir_tp+=tp; dir_fp+=fp; dir_fn+=fn
            per_f1_dir.append(f1)
        else:
            indir_tp+=tp; indir_fp+=fp; indir_fn+=fn

    def micro(tp,fp,fn):
        p=tp/(tp+fp) if tp+fp else 0.0
        r=tp/(tp+fn) if tp+fn else 0.0
        return p,r,(2*p*r/(p+r) if p+r else 0.0)

    mic_p,mic_r,mic_f1 = micro(all_tp,all_fp,all_fn)
    _,_,indir_f1    = micro(indir_tp,indir_fp,indir_fn)
    macro_f1        = sum(per_f1)/len(per_f1) if per_f1 else 0.0
    direct_macro_f1 = sum(per_f1_dir)/len(per_f1_dir) if per_f1_dir else 0.0
    composite       = ALPHA*direct_macro_f1 + (1-ALPHA)*indir_f1

    return {
      "micro":             {"precision":mic_p, "recall":mic_r, "f1":mic_f1},
      "macro_f1":          macro_f1,
      "direct_macro_f1":   direct_macro_f1,
      "indirect_micro_f1": indir_f1,
      "composite":         composite,
    }

def main():
    gold = load_jsonl(GOLD_FILE)
    print(f"Loaded {len(gold)} gold examples.\n")
    for name,path in PRED_FILES.items():
        preds = load_jsonl(path)
        stats = evaluate_model(gold, preds)
        print(f"=== {name} ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False), "\n")

if __name__=="__main__":
    main()
