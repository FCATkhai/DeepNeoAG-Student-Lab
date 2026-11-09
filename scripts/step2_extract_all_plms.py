from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
import os
import glob
import gc
import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel

os.chdir(str(ROOT_DIR))

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

SPLITS = [
    ("Train/pos", ROOT_DIR / "fasta" / "Train" / "pos"),
    ("Train/neg", ROOT_DIR / "fasta" / "Train" / "neg"),
    ("Test/pos",  ROOT_DIR / "fasta" / "Test" / "pos"),
    ("Test/neg",  ROOT_DIR / "fasta" / "Test" / "neg"),
    ("Temporal_pos", ROOT_DIR / "fasta" / "Temporal" / "pos"),
    ("Temporal_neg", ROOT_DIR / "fasta" / "Temporal" / "neg"),
]

def read_all_fastas(folder):
    files = sorted(glob.glob(os.path.join(str(folder), "*.fasta")))
    seqs = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        seq_id = os.path.splitext(os.path.basename(fpath))[0]
        seq = "".join([l for l in lines if not l.startswith(">")])
        seqs.append((seq_id, seq))
    return seqs

def pad_or_truncate(emb_2d, maxseq=35, feat_dim=1024):
    arr = np.zeros((maxseq, feat_dim), dtype=np.float16)
    dlen = len(emb_2d)
    if dlen < maxseq:
        arr[:dlen, :] = emb_2d
    else:
        arr[:, :] = emb_2d[:maxseq, :]
    return arr.reshape((1, 1, maxseq, feat_dim))

def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

def empty_device_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

# -------------------------------------------------------------------------
# 1. Biophysical + Immunogenicity + BLOSUM62 Feature Extractor (ImmunoPhys-48)
# -------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Standard BLOSUM62 matrix rows in AA_ORDER
BLOSUM62 = {
    'A': [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    'C': [-1, 9,-3,-4,-2,-3,-3,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    'D': [-2,-3, 6, 2,-3,-1,-1,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    'E': [-2,-4, 2, 5,-3,-2, 0,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    'F': [ 0,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3, 0, 3,-4,-2,-2, 1, 3,-1],
    'G': [-1,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-3,-2, 0,-2,-2,-3,-3],
    'H': [-1,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    'I': [ 0,-3,-1,-2, 0,-4,-3, 4,-3, 2, 1,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    'K': [-2,-3,-1, 0,-3,-2,-1,-3, 5,-2,-3, 0,-1,-3,-1, 0,-1,-3,-2,-2],
    'L': [-1,-1,-3,-3, 0,-4,-3, 2,-2, 4, 2,-3, 2, 0,-3,-2,-1,-2,-1, 1],
    'M': [-1,-1,-4,-3, 0,-3,-2, 1,-3, 2, 5,-2, 0, 0,-2,-1,-1,-1,-1, 1],
    'N': [-1,-3,-1, 1,-3, 0,-1,-3, 0,-3,-2, 6,-2,-3,-2, 1, 0,-4,-2,-3],
    'P': [-1,-1,-3,-2, 0,-2,-2, 1,-1, 2, 0,-2, 7,-1,-2,-1,-1,-4,-3,-2],
    'Q': [-2,-2,-3,-3, 3,-3,-1, 0,-3, 0, 0,-3,-1, 5,-1, 0,-1,-2,-1,-2],
    'R': [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2,-2,-1, 5,-1,-1,-3,-2,-3],
    'S': [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-3,-2,-2],
    'T': [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5,-2,-2, 0],
    'V': [-3,-2,-4,-3, 1,-2,-2, 3,-3, 1, 1,-4,-4,-2,-3,-3,-2,11, 2,-3],
    'W': [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2, 2, 7,-1],
    'Y': [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0,-3,-1, 4],
}

# Calis et al. (PLOS Comp Biol) T-cell immunogenicity log-enrichment score
CALIS_IMMUNO = {
    'W': 0.342, 'F': 0.289, 'Y': 0.145, 'I': 0.168, 'L': -0.015,
    'E': 0.185, 'R': 0.088, 'H': 0.062, 'T': 0.042, 'V': 0.025,
    'P': 0.010, 'A': -0.028, 'G': -0.055, 'N': -0.082, 'D': -0.095,
    'C': -0.110, 'Q': -0.165, 'M': -0.210, 'S': -0.245, 'K': -0.275,
}
# Kyte-Doolittle Hydrophobicity (scaled / 4.5)
KD_HYDRO = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2,
    'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}
# Zimmerman Bulkiness (scaled / 21.6)
BULKINESS = {
    'W': 21.67, 'Y': 18.03, 'F': 19.80, 'I': 21.40, 'L': 21.40, 'V': 21.57,
    'M': 16.25, 'R': 14.28, 'K': 15.71, 'H': 13.69, 'P': 17.43, 'E': 13.57,
    'Q': 14.45, 'D': 11.68, 'N': 12.82, 'T': 15.77, 'S': 9.47, 'C': 13.46,
    'A': 11.50, 'G': 3.40,
}
# Net charge at pH 7.4
CHARGE = {
    'R': 1.0, 'K': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0,
}
# Molecular weight (scaled / 204.2)
MW = {
    'G': 75.1, 'A': 89.1, 'S': 105.1, 'P': 115.1, 'V': 117.1, 'T': 119.1,
    'C': 121.2, 'I': 131.2, 'L': 131.2, 'N': 132.1, 'D': 133.1, 'Q': 146.2,
    'K': 146.2, 'E': 147.1, 'M': 149.2, 'H': 155.2, 'F': 165.2, 'R': 174.2,
    'Y': 181.2, 'W': 204.2,
}

def compute_immunophys_seq(seq):
    L = len(seq)
    mat = np.zeros((L, 48), dtype=np.float32)
    for p, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa, 0)
        # 0..19: One-hot
        mat[p, idx] = 1.0
        # 20..39: BLOSUM62 / 5.0
        if aa in BLOSUM62:
            mat[p, 20:40] = np.array(BLOSUM62[aa], dtype=np.float32) / 5.0
        # 40..47: 8 Immunophysical & positional descriptors
        mat[p, 40] = CALIS_IMMUNO.get(aa, 0.0) * 3.0
        mat[p, 41] = KD_HYDRO.get(aa, 0.0) / 4.5
        mat[p, 42] = BULKINESS.get(aa, 15.0) / 21.67
        mat[p, 43] = CHARGE.get(aa, 0.0)
        mat[p, 44] = 1.0 if aa in "FWYH" else 0.0  # Aromaticity
        mat[p, 45] = MW.get(aa, 130.0) / 204.2
        rel_pos = (p + 1) / max(L, 1)
        mat[p, 46] = rel_pos
        # TCR central bulge contact weight (peaks at center of peptide P4-P8)
        mat[p, 47] = np.exp(-0.5 * ((rel_pos - 0.55) / 0.22) ** 2)
    return mat

def extract_immunophys():
    print("\n[1/5] Extracting ImmunoPhys-48 (d=48) ...", flush=True)
    t0 = time.perf_counter()
    for split_name, folder in SPLITS:
        out_path = ROOT_DIR / "dataset" / "immunophys" / f"{split_name}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seqs = read_all_fastas(folder)
        tensor_list = [pad_or_truncate(compute_immunophys_seq(s), 35, 48) for _, s in seqs]
        np.save(str(out_path), np.concatenate(tensor_list, axis=0))
    print(f"  ==> ImmunoPhys-48 done in {time.perf_counter() - t0:.2f}s", flush=True)

# -------------------------------------------------------------------------
# 2. ESM-1v (650M) - facebook/esm1v_t33_650M_UR90S_1 (d=1280)
# -------------------------------------------------------------------------
def extract_hf_esm(model_repo, save_key, feat_dim, device_str=DEFAULT_DEVICE, batch_size=64):
    device = torch.device(device_str)
    print(f"\nExtracting [{save_key}] ({model_repo}, d={feat_dim}) on {device} ...", flush=True)
    t_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_repo, trust_remote_code=True)
    if device.type == "mps":
        model = model.half()
    model = model.to(device).eval()

    timings = {}
    for split_name, folder in SPLITS:
        out_path = ROOT_DIR / "dataset" / save_key / f"{split_name}.npy"
        if out_path.exists():
            print(f"  [{save_key}] {split_name} already exists, skipping.")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seqs = read_all_fastas(folder)
        t0 = time.perf_counter()
        # Sort by length for minimal padding
        seq_sorted = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
        split_embs = {}
        with torch.no_grad():
            for i in range(0, len(seq_sorted), batch_size):
                batch = seq_sorted[i : i + batch_size]
                ids_b = [b[0] for b in batch]
                strs_b = [b[1] for b in batch]
                enc = tokenizer(strs_b, add_special_tokens=True, padding="longest", return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state.detach().cpu().numpy()
                for b_i, seq_id in enumerate(ids_b):
                    L = len(strs_b[b_i])
                    split_embs[seq_id] = out[b_i, 1 : 1 + L, :]
        sync_device(device)
        dt = time.perf_counter() - t0
        timings[split_name] = dt
        tensor_list = [pad_or_truncate(split_embs[sid], 35, feat_dim) for sid, _ in seqs]
        np.save(str(out_path), np.concatenate(tensor_list, axis=0))
        print(f"  [{save_key}] {split_name:14s} ({len(seqs):4d} seqs): {dt:6.2f}s", flush=True)

    del model, tokenizer
    gc.collect()
    empty_device_cache(device)
    print(f"  ==> [{save_key}] Total time: {time.perf_counter() - t_start:.2f}s", flush=True)
    return timings

# -------------------------------------------------------------------------
# 3. ProstT5 - Rostlab/ProstT5 (d=1024, <AA2fold> prefix)
# -------------------------------------------------------------------------
def extract_prostt5(device_str=DEFAULT_DEVICE, batch_size=64):
    device = torch.device(device_str)
    save_key = "prostt5"
    model_repo = "Rostlab/ProstT5"
    feat_dim = 1024
    print(f"\nExtracting [{save_key}] ({model_repo}, d={feat_dim}) on {device} ...", flush=True)
    t_start = time.perf_counter()
    tokenizer = T5Tokenizer.from_pretrained(model_repo, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_repo)
    if device.type == "mps":
        model = model.half()
    model = model.to(device).eval()

    timings = {}
    for split_name, folder in SPLITS:
        out_path = ROOT_DIR / "dataset" / save_key / f"{split_name}.npy"
        if out_path.exists():
            print(f"  [{save_key}] {split_name} already exists, skipping.")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seqs = read_all_fastas(folder)
        t0 = time.perf_counter()
        seq_sorted = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
        split_embs = {}
        with torch.no_grad():
            for i in range(0, len(seq_sorted), batch_size):
                batch = seq_sorted[i : i + batch_size]
                ids_b = [b[0] for b in batch]
                strs_b = [b[1] for b in batch]
                # ProstT5 requires "<AA2fold> " prefix + space-separated uppercase AAs
                prefixed_b = ["<AA2fold> " + " ".join(list(s)) for s in strs_b]
                enc = tokenizer(prefixed_b, add_special_tokens=True, padding="longest", return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state.detach().cpu().numpy()
                for b_i, seq_id in enumerate(ids_b):
                    L = len(strs_b[b_i])
                    # Index 0 is <AA2fold>, indices 1..L are the amino acids
                    split_embs[seq_id] = out[b_i, 1 : 1 + L, :]
        sync_device(device)
        dt = time.perf_counter() - t0
        timings[split_name] = dt
        tensor_list = [pad_or_truncate(split_embs[sid], 35, feat_dim) for sid, _ in seqs]
        np.save(str(out_path), np.concatenate(tensor_list, axis=0))
        print(f"  [{save_key}] {split_name:14s} ({len(seqs):4d} seqs): {dt:6.2f}s", flush=True)

    del model, tokenizer
    gc.collect()
    empty_device_cache(device)
    print(f"  ==> [{save_key}] Total time: {time.perf_counter() - t_start:.2f}s", flush=True)
    return timings

# -------------------------------------------------------------------------
# 4. Ankh-Base - ElnaggarLab/ankh-base (d=768, exact vocab ID mapping)
# -------------------------------------------------------------------------
def extract_ankh(device_str=DEFAULT_DEVICE, batch_size=64):
    device = torch.device(device_str)
    save_key = "ankh_base"
    model_repo = "ElnaggarLab/ankh-base"
    feat_dim = 768
    print(f"\nExtracting [{save_key}] ({model_repo}, d={feat_dim}) on {device} ...", flush=True)
    t_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    vocab = tokenizer.get_vocab()
    pad_id = vocab["<pad>"]  # 0
    eos_id = vocab["</s>"]   # 1
    unk_id = vocab["<unk>"]  # 2

    model = T5EncoderModel.from_pretrained(model_repo)
    # T5 relative position bias works best in float32 on MPS for Ankh
    model = model.to(torch.float32).to(device).eval()

    timings = {}
    for split_name, folder in SPLITS:
        out_path = ROOT_DIR / "dataset" / save_key / f"{split_name}.npy"
        if out_path.exists():
            print(f"  [{save_key}] {split_name} already exists, skipping.")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seqs = read_all_fastas(folder)
        t0 = time.perf_counter()
        seq_sorted = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
        split_embs = {}
        with torch.no_grad():
            for i in range(0, len(seq_sorted), batch_size):
                batch = seq_sorted[i : i + batch_size]
                ids_b = [b[0] for b in batch]
                strs_b = [b[1] for b in batch]
                max_l = max(len(s) for s in strs_b) + 1  # +1 for </s>
                input_ids = np.full((len(batch), max_l), pad_id, dtype=np.int64)
                attn_mask = np.zeros((len(batch), max_l), dtype=np.int64)
                for b_i, s in enumerate(strs_b):
                    toks = [vocab.get(aa, unk_id) for aa in s] + [eos_id]
                    input_ids[b_i, :len(toks)] = toks
                    attn_mask[b_i, :len(toks)] = 1
                input_ids_t = torch.tensor(input_ids, device=device)
                attn_mask_t = torch.tensor(attn_mask, device=device)
                out = model(input_ids=input_ids_t, attention_mask=attn_mask_t).last_hidden_state.detach().cpu().numpy()
                for b_i, seq_id in enumerate(ids_b):
                    L = len(strs_b[b_i])
                    split_embs[seq_id] = out[b_i, :L, :]
        sync_device(device)
        dt = time.perf_counter() - t0
        timings[split_name] = dt
        tensor_list = [pad_or_truncate(split_embs[sid], 35, feat_dim) for sid, _ in seqs]
        np.save(str(out_path), np.concatenate(tensor_list, axis=0))
        print(f"  [{save_key}] {split_name:14s} ({len(seqs):4d} seqs): {dt:6.2f}s", flush=True)

    del model, tokenizer
    gc.collect()
    empty_device_cache(device)
    print(f"  ==> [{save_key}] Total time: {time.perf_counter() - t_start:.2f}s", flush=True)
    return timings


def extract_tape(device_str=DEFAULT_DEVICE, batch_size=64):
    from collections import defaultdict
    from tape import ProteinBertModel, TAPETokenizer
    device = torch.device(device_str)
    save_key = "tape"
    print(f"\nExtracting [{save_key}] (TAPE bert-base, d=768) on {device} ...", flush=True)
    model = ProteinBertModel.from_pretrained("bert-base").to(device).eval()
    tokenizer = TAPETokenizer(vocab="iupac")
    for split_name, folder in SPLITS:
        out_path = ROOT_DIR / "dataset" / save_key / f"{split_name}.npy"
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seqs = read_all_fastas(folder)
        length_groups = defaultdict(list)
        for seq_id, seq in seqs:
            enc = tokenizer.encode(seq)
            length_groups[len(enc)].append((seq_id, enc))
        split_embs = {}
        with torch.no_grad():
            for tok_len, items in sorted(length_groups.items()):
                for idx in range(0, len(items), batch_size):
                    batch = items[idx : idx + batch_size]
                    token_ids = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long, device=device)
                    out = model(token_ids)[0][:, 1:-1, :].detach().cpu().numpy()
                    for i, (sid, _) in enumerate(batch):
                        split_embs[sid] = out[i]
        tensor_list = [pad_or_truncate(split_embs[sid], 35, 768) for sid, _ in seqs]
        np.save(str(out_path), np.concatenate(tensor_list, axis=0))
    del model, tokenizer
    gc.collect()
    empty_device_cache(device)

def extract_prottrans(device_str=DEFAULT_DEVICE, batch_size=64):
    device = torch.device(device_str)
    save_key = "ProtTrans"
    model_repo = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"\nExtracting [{save_key}] ({model_repo}, d={1024}) on {device} ...", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(model_repo, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_repo)
    if device.type in ("mps", "cuda"):
        model = model.half()
    model = model.to(device).eval()
    for split_name, folder in SPLITS:
        out_path = ROOT_DIR / "dataset" / save_key / f"{split_name}.npy"
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seqs = read_all_fastas(folder)
        seq_sorted = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
        split_embs = {}
        with torch.no_grad():
            for i in range(0, len(seq_sorted), batch_size):
                batch = seq_sorted[i : i + batch_size]
                ids_b = [b[0] for b in batch]
                strs_b = [b[1] for b in batch]
                spaced_b = [" ".join(list(s)) for s in strs_b]
                enc = tokenizer(spaced_b, add_special_tokens=True, padding="longest", return_tensors="pt")
                out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).last_hidden_state.detach().cpu().numpy()
                for b_i, sid in enumerate(ids_b):
                    L = len(strs_b[b_i])
                    split_embs[sid] = out[b_i, :L, :]
        tensor_list = [pad_or_truncate(split_embs[sid], 35, 1024) for sid, _ in seqs]
        np.save(str(out_path), np.concatenate(tensor_list, axis=0))
    del model, tokenizer
    gc.collect()
    empty_device_cache(device)

if __name__ == "__main__":
    dev_str = DEFAULT_DEVICE
    all_timings = {}
    extract_immunophys()
    extract_tape(dev_str, 64)
    extract_prottrans(dev_str, 64)
    all_timings["esm2"] = extract_hf_esm("facebook/esm2_t33_650M_UR50D", "esm2", 1280, dev_str, 64)
    all_timings["esmc_300m"] = extract_hf_esm("Synthyra/ESMplusplus_small", "esmc_300m", 960, dev_str, 64)
    all_timings["ankh_base"] = extract_ankh(dev_str, 64)
    all_timings["esm1v"] = extract_hf_esm("facebook/esm1v_t33_650M_UR90S_1", "esm1v", 1280, dev_str, 64)
    all_timings["prostt5"] = extract_prostt5(dev_str, 64)
    timing_file = ROOT_DIR / "data" / "new_plm_extraction_timings.json"
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_file, "w", encoding="utf-8") as f:
        json.dump(all_timings, f, indent=2)
    print("\nAll new PLM embeddings extracted and saved successfully!", flush=True)
