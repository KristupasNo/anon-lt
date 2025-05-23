from transformers import AutoTokenizer, EvalPrediction
from sklearn.metrics import precision_recall_fscore_support
import evaluate

tokenizer = AutoTokenizer.from_pretrained(
    "neurotechnology/Lt-Llama-2-13b-instruct-hf", use_fast=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred: EvalPrediction):
    pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids

    decode_label_ids = [
        [tok if tok != -100 else tokenizer.pad_token_id for tok in seq]
        for seq in label_ids
    ]

    preds  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    labels = tokenizer.batch_decode(decode_label_ids, skip_special_tokens=True)

    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    exact = sum(p_.strip() == l_.strip() for p_, l_ in zip(preds, labels)) / len(preds)

    rouge_scores = rouge.compute(predictions=preds, references=labels)
    rouge1 = rouge_scores["rouge1"]
    rougel = rouge_scores["rougeL"]

    return {
        "precision":   p,
        "recall":      r,
        "f1":          f,
        "exact_match": exact,
        "rouge1":      rouge1,
        "rougeL":      rougel,
    }