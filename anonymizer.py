#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

CACHE_DIR = "/workspace/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_MODEL = "neurotechnology/Lt-Llama-2-13b-instruct-hf"
LORA_WEIGHTS = "krinor/llama2-anon-lt"

print(f"Loading tokenizer to cache: {CACHE_DIR}")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=False,
    cache_dir=CACHE_DIR,
    token=os.environ.get("HF_TOKEN", None),
)

print("Loading base model with 8-bit quantization...")
quant_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
    cache_dir=CACHE_DIR,
    token=os.environ.get("HF_TOKEN", None),
)

print("Loading LoRA weights...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS,
    device_map="auto",
    cache_dir=CACHE_DIR,
    token=os.environ.get("HF_TOKEN", None),
)
model.eval()

model.config.max_length = 4096
model.config.eos_token_id = tokenizer.eos_token_id

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "Esi paslaugus asistentas – anonimizuok visus tiesioginius ir netiesioginius identifikatorius.\n"
    "<</SYS>>\n"
    "{instruction}\n"
    "{input_text} [/INST]\n"
)

def anonymize(instruction: str, text: str, max_new_tokens: int = 512):
    prompt = PROMPT_TEMPLATE.format_map({
        "instruction": instruction,
        "input_text": text
    })
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()
              if k in ("input_ids","attention_mask")}

    prompt_length = inputs["input_ids"].shape[1]
    total_max_length = prompt_length + max_new_tokens

    outputs = model.generate(
        **inputs,
        max_length=total_max_length,
        num_beams=4,
        early_stopping=True,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("[/INST]")[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="Anonimizuok REPL/CLI")
    parser.add_argument("-i", "--instruction", help="Anonimizavimo instrukcija")
    parser.add_argument("-t", "--input",       help="Tekstas anonimizavimui")
    args = parser.parse_args()

    if args.instruction and args.input:
        print(anonymize(args.instruction, args.input))
    else:
        print("Anonimizuok REPL (Ctrl-C to exit).")
        try:
            while True:
                instr = input("instruction: ").strip()
                if not instr:
                    continue
                txt = input("input: ").strip()
                print("\n" + anonymize(instr, txt) + "\n")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")

if __name__ == "__main__":
    main()