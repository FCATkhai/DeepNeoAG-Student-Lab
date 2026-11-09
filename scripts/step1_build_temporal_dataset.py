from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
import json
import glob
import os
from collections import Counter
import numpy as np

os.chdir(str(ROOT_DIR))

data_file = ROOT_DIR / "data" / "iedb_new_melanoma_epitopes.json"
with open(data_file, "r", encoding="utf-8") as f:
    rows = json.load(f)

# Load original DeepNeoAG sequences (Train + Test)
orig_pos_train, orig_pos_test = {}, {}
orig_neg_train, orig_neg_test = {}, {}
for folder, d in [
    (str(ROOT_DIR / "fasta" / "Train" / "pos"), orig_pos_train),
    (str(ROOT_DIR / "fasta" / "Test" / "pos"), orig_pos_test),
    (str(ROOT_DIR / "fasta" / "Train" / "neg"), orig_neg_train),
    (str(ROOT_DIR / "fasta" / "Test" / "neg"), orig_neg_test),
]:
    for fp in sorted(glob.glob(os.path.join(folder, "*.fasta"))):
        with open(fp) as f:
            seq = "".join([l.strip() for l in f if l.strip() and not l.startswith(">")])
            d[seq] = os.path.basename(fp)

orig_all_seqs = list(set(orig_pos_train) | set(orig_pos_test) | set(orig_neg_train) | set(orig_neg_test))
orig_all_set = set(orig_all_seqs)
print(f"Original DeepNeoAG total unique sequences: {len(orig_all_set)}")

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

new_neo_records = []
new_taa_records = []

seen_neo = set()
seen_taa = set()

for r in rows:
    seq = (r.get("linear_sequence") or "").strip().upper()
    if not seq or not set(seq).issubset(STD_AA):
        continue
    if len(seq) >= 35 or len(seq) < 8:
        continue
    if seq in orig_all_set:
        continue
    if r.get("e_modification") is not None:
        continue
    if "NCBITaxon:9606" not in str(r.get("host_organism_iris", "")):
        continue
    quals = r.get("qualitative_measures") or []
    if not any(q.startswith("Positive") for q in quals):
        continue

    ys = [int(y) for y in (r.get("reference_dates") or []) if str(y).isdigit()]
    min_year = min(ys) if ys else 2024
    max_year = max(ys) if ys else 2024
    mhc_c = r.get("mhc_classes") or []

    rel = r.get("e_related_object_types")
    if rel and any("neo-epitope" in t for t in rel):
        if "NCBITaxon:9606" in str(r.get("r_object_source_organism_iris", "")):
            if seq not in seen_neo:
                seen_neo.add(seq)
                r["_min_year"] = min_year
                r["_max_year"] = max_year
                r["_mhc_classes"] = mhc_c
                new_neo_records.append(r)
    elif rel is None:
        if "NCBITaxon:9606" in str(r.get("source_organism_iris", "")):
            if seq not in seen_taa:
                seen_taa.add(seq)
                r["_min_year"] = min_year
                r["_max_year"] = max_year
                r["_mhc_classes"] = mhc_c
                new_taa_records.append(r)

# Remove any overlap between new_neo and new_taa
neo_seq_set = {r["linear_sequence"] for r in new_neo_records}
new_taa_records = [r for r in new_taa_records if r["linear_sequence"] not in neo_seq_set]

print(f"Dataset T1 (All Newly Added, Deduplicated, Length 8..34): Neoantigens={len(new_neo_records)}, TAAs={len(new_taa_records)}")

# Sequence identity calculation (maximum ungapped/sliding alignment identity over shorter sequence length, matching short-peptide CD-HIT / sequence similarity definition)
def max_seq_identity(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    l1, l2 = len(s1), len(s2)
    if l1 == 0:
        return 0.0
    best_matches = 0
    # Slide s1 across s2 including overhangs of at least 5 aa
    for offset in range(-l1 + 5, l2 - 4):
        matches = 0
        for i in range(l1):
            j = i + offset
            if 0 <= j < l2 and s1[i] == s2[j]:
                matches += 1
        if matches > best_matches:
            best_matches = matches
    return best_matches / l1

def filter_similarity_30(records, reference_seqs, threshold=0.30):
    kept = []
    pool = list(reference_seqs)
    for r in records:
        seq = r["linear_sequence"]
        too_similar = False
        # First check against already kept sequences in this new class and original dataset
        for ref in pool:
            if max_seq_identity(seq, ref) >= threshold:
                too_similar = True
                break
        if not too_similar:
            kept.append(r)
            pool.append(seq)
    return kept

# Let us check both:
# (a) Internal <30% similarity within new_neo and new_taa (plus exact exclusion of original dataset, or <30% against original same-class)
orig_pos_seqs = list(set(orig_pos_train) | set(orig_pos_test))
orig_neg_seqs = list(set(orig_neg_train) | set(orig_neg_test))

neo_cdhit_internal = filter_similarity_30(new_neo_records, [], threshold=0.30)
taa_cdhit_internal = filter_similarity_30(new_taa_records, [], threshold=0.30)
print(f"Internal <30% similarity (Paper protocol on new data): Neoantigens={len(neo_cdhit_internal)}, TAAs={len(taa_cdhit_internal)}")

neo_cdhit_strict = filter_similarity_30(new_neo_records, orig_pos_seqs, threshold=0.30)
taa_cdhit_strict = filter_similarity_30(new_taa_records, orig_neg_seqs, threshold=0.30)
print(f"Strict <30% similarity (also <30% vs original DeepNeoAG): Neoantigens={len(neo_cdhit_strict)}, TAAs={len(taa_cdhit_strict)}")

# Save all processed datasets for feature extraction and evaluation
(ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)
out_dataset_file = ROOT_DIR / "data" / "temporal_datasets.json"
with open(out_dataset_file, "w", encoding="utf-8") as f:
    json.dump({
        "T1_all_new": {
            "pos": new_neo_records,
            "neg": new_taa_records,
        },
        "T2_cdhit30_internal": {
            "pos": neo_cdhit_internal,
            "neg": taa_cdhit_internal,
        },
        "T3_cdhit30_strict_vs_orig": {
            "pos": neo_cdhit_strict,
            "neg": taa_cdhit_strict,
        }
    }, f, indent=2)
print("Saved temporal_datasets.json successfully!")
