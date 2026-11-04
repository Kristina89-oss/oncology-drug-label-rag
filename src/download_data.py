"""Download FDA drug labels (public domain) from openFDA into data/raw/*.json."""
import json
import pathlib
import time

import requests

RAW = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
API = "https://api.fda.gov/drug/label.json"

# Oncology-focused set: matches the author's domain and gives dense dosing/safety text.
DRUGS = [
    "abiraterone", "enzalutamide", "apalutamide", "darolutamide",
    "docetaxel", "cabazitaxel", "leuprolide", "degarelix",
    "tamoxifen", "anastrozole", "letrozole", "exemestane",
    "capecitabine", "cisplatin", "carboplatin", "oxaliplatin",
    "paclitaxel", "gemcitabine", "olaparib", "pembrolizumab",
]

SECTIONS = [
    "indications_and_usage", "dosage_and_administration", "contraindications",
    "warnings_and_cautions", "warnings", "adverse_reactions", "drug_interactions",
    "use_in_specific_populations", "overdosage", "clinical_pharmacology",
]


def fetch(name: str):
    params = {"search": f'openfda.generic_name:"{name}"', "limit": 1}
    r = requests.get(API, params=params, timeout=60)
    if r.status_code != 200:
        return None
    results = r.json().get("results") or []
    return results[0] if results else None


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in DRUGS:
        out = RAW / f"{name}.json"
        if out.exists():
            continue
        rec = fetch(name)
        if not rec:
            print("MISSING", name)
            continue
        doc = {
            "drug": name,
            "set_id": rec.get("set_id"),
            "effective_time": rec.get("effective_time"),
            "brand": (rec.get("openfda", {}).get("brand_name") or [""])[0],
            "sections": {s: " ".join(rec[s]) for s in SECTIONS if rec.get(s)},
        }
        out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print("OK", name, len(doc["sections"]), "sections")
        time.sleep(0.4)


if __name__ == "__main__":
    main()
