#!/usr/bin/env python3
import json
import io
from tqdm import tqdm
from anonymizer import anonymize

GOLD_FILE   = "gold.jsonl"
OUTPUT_FILE = "lt_llama.jsonl"

LEGAL_EXAMPLES = (
    "1. \"Advokatė Lina iš Šilalės konsultuoja vietos gyventojus paveldėjimo klausimais. "
    "Ji neseniai padėjo parengti skundo projektą, susijusį su testamento galiojimu.\" → "
    "\"[Profesija] [Vardas] iš [Miestas] konsultuoja [Šeimyninė_informacija] [Dokumentas]. "
    "Ji neseniai padėjo parengti [Dokumentas], susijusį su [Dokumentas] galiojimu.\"\n"
    "2. \"Savivaldybės teisininkas Andrius Juknevičius prieš mėnesį pateikė skundą administraciniam teismui. "
    "Jis atstovavo vienai iš statybų bendrovių ir ginčijo sklypo paskirties pakeitimą. "
    "Pateikti dokumentai buvo analizuojami tris savaites.\" → "
    "\"[Profesija] [Vardas_Pavardė] prieš [Laikotarpis] pateikė skundą [Įstaiga]. "
    "Jis atstovavo vienai iš [Įstaiga] ir ginčijo [Dokumentas]. "
    "Pateikti [Dokumentas] buvo analizuojami tris [Laikotarpis].\"\n\n"
    "Dabar anonimizuokite: \"{input_text}\""
)

INSTR_MED = (
    "Anonimizuokite visus tiesioginius ir netiesioginius identifikatorius, "
    "galinčius padėti atskleisti paciento tapatybę."
)
INSTR_LEG = (
    "Remdamiesi žemiau pateiktais anonimizavimo pavyzdžiais, "
    "anonimizuokite asmens duomenis pateiktame tekste."
)
INSTR_GOV = (
    "Pirmiausia atpažinkite pateiktame tekste esančius tiesioginius ir netiesioginius identifikatorius "
    "pagal žemiau pateiktą žymeklių sąrašą, tada anonimizuokite tekstą, pakeisdami kiekvieną "
    "identifikatorių atitinkamu žymekliu."
)

def prepare_input(bucket: str, raw: str) -> (str, str):
    """
    Returns (instruction, input_text) for this example.
    """
    if bucket.startswith(("B-ZD-Med", "B-ZI-Med")):
        return INSTR_MED, raw

    elif bucket.startswith(("B-FD-Leg", "B-FI-Leg")):
        inp = LEGAL_EXAMPLES.format(input_text=raw)
        return INSTR_LEG, inp

    elif bucket.startswith(("B-CD-Gov", "B-CI-Gov")):
        return INSTR_GOV, raw

    else:
        return INSTR_MED, raw

def main():
    results = []

    with open(GOLD_FILE, encoding="utf-8") as f:
        buf = f.read(1)
        while buf and buf.isspace():
            buf = f.read(1)
        f.seek(0)
        if buf == "[":
            all_ex = json.load(f)
        else:
            all_ex = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_ex.append(json.loads(line))

    for ex in tqdm(all_ex, desc="Reading gold"):
        instr, inp = prepare_input(ex["bucket"], ex["raw"])
        pred = anonymize(instr, inp, max_new_tokens=2048)
        results.append({
            "id":   ex["id"],
            "pred": pred
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
