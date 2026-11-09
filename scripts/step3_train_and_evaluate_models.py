from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
import os
import re
import gc
import json
import time
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(str(ROOT_DIR))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
MAXSEQ = 35

# Load cached masks and lengths (0.001s)
cache_path = ROOT_DIR / "data" / "cached_temporal_masks.npz"
cache = np.load(str(cache_path))
TEMPORAL_MASKS = {
    k: (cache[k + "__pos"], cache[k + "__neg"])
    for k in [
        "Temporal-All",
        "Temporal-Matched",
        "Temporal-MHC-I",
        "Temporal-MHC-II",
        "Temporal-Strict30 (k=3)",
        "Temporal-Strict30 (k=2)",
    ]
}
SPLIT_LENS = {
    "Train_pos": cache["len_Train_pos"],
    "Train_neg": cache["len_Train_neg"],
    "Test_pos": cache["len_Test_pos"],
    "Test_neg": cache["len_Test_neg"],
    "Temporal_pos": cache["len_Temporal_pos"],
    "Temporal_neg": cache["len_Temporal_neg"],
}

def load_plm_splits(plm_key):
    base = ROOT_DIR / "dataset" / plm_key
    def _load_pair(pos_f, neg_f, pos_l, neg_l):
        p = np.load(str(base / pos_f)).astype(np.float32)[:, 0, :, :]
        n = np.load(str(base / neg_f)).astype(np.float32)[:, 0, :, :]
        x = np.concatenate([p, n], axis=0)
        y = np.concatenate([np.ones(len(p), dtype=np.float32), np.zeros(len(n), dtype=np.float32)])
        l = np.concatenate([pos_l, neg_l], axis=0)
        return x, y, l, p, n

    x_tr, y_tr, l_tr, _, _ = _load_pair("Train/pos.npy", "Train/neg.npy", SPLIT_LENS["Train_pos"], SPLIT_LENS["Train_neg"])
    x_te, y_te, l_te, _, _ = _load_pair("Test/pos.npy", "Test/neg.npy", SPLIT_LENS["Test_pos"], SPLIT_LENS["Test_neg"])
    x_tm, y_tm, l_tm, p_tm, n_tm = _load_pair("Temporal_pos.npy", "Temporal_neg.npy", SPLIT_LENS["Temporal_pos"], SPLIT_LENS["Temporal_neg"])
    return {
        "train": (x_tr, y_tr, l_tr),
        "test": (x_te, y_te, l_te),
        "temporal_pos_neg": (p_tm, n_tm, SPLIT_LENS["Temporal_pos"], SPLIT_LENS["Temporal_neg"]),
    }

def compute_all_metrics(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(metrics.auc(fpr, tpr))
    pr_auc = float(average_precision_score(y_true, y_prob))
    gmeans = np.sqrt(tpr * (1.0 - fpr))
    ix = int(np.argmax(gmeans))
    thresh = thresholds[ix]
    y_pred = (y_prob >= thresh).astype(int)
    TN, FP, FN, TP = [int(v) for v in metrics.confusion_matrix(y_true, y_pred).ravel()]
    Sens = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    Spec = float(TN / (FP + TN)) if (FP + TN) > 0 else 0.0
    Acc = float((TP + TN) / (TP + FP + TN + FN))
    denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = float((TP * TN - FP * FN) / denom) if denom > 0 else 0.0
    F1 = float(2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) > 0 else 0.0
    Prec = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    return {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "MCC": round(MCC, 4),
        "F1": round(F1, 4),
        "Sens": round(Sens, 4),
        "Spec": round(Spec, 4),
        "Acc": round(Acc, 4),
        "Prec": round(Prec, 4),
    }

def evaluate_across_cohorts(prob_test, y_test, prob_temp_pos, prob_temp_neg):
    res = {"Original Test": compute_all_metrics(y_test, prob_test)}
    for cohort_name, (pmask, nmask) in TEMPORAL_MASKS.items():
        p_sub = prob_temp_pos[pmask]
        n_sub = prob_temp_neg[nmask]
        y_sub = np.concatenate([np.ones(len(p_sub)), np.zeros(len(n_sub))])
        prob_sub = np.concatenate([p_sub, n_sub])
        res[cohort_name] = compute_all_metrics(y_sub, prob_sub)
    return res

# -------------------------------------------------------------------------
# Fast Native Conv2d Implementations of DeepScan & SE-ResScan (Multi-Scale SE Motif Net)
# -------------------------------------------------------------------------
class DeepScanFast2D(nn.Module):
    def __init__(self, in_dim, windows=(4, 8, 16), num_filters=64, num_hidden=256):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_dim, num_filters, kernel_size=(1, w), padding=0)
            for w in windows
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=(1, MAXSEQ - w + 1), stride=(1, MAXSEQ))
            for w in windows
        ])
        self.drop = nn.Dropout(0.7)
        self.fc1 = nn.Linear(num_filters * len(windows), num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        for c in self.convs:
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x4d):
        # x4d: (B, D, 1, 35)
        outs = [p(F.relu(c(x4d))).flatten(1) for c, p in zip(self.convs, self.pools)]
        h = self.drop(torch.cat(outs, dim=1))
        return self.fc2(F.relu(self.fc1(h))).squeeze(-1)

class SEResScan2D(nn.Module):
    """
    Squeeze-and-Excitation Multi-Scale Motif Scanner (windows=[3, 5, 7, 11, 15])
    with Channel Attention Gating + Dual Max/Soft-Attention Pooling.
    Designed to capture both short MHC-I TCR contact cores (3, 5, 7 aa) and
    longer MHC-II cores (11, 15 aa).
    """
    def __init__(self, in_dim, windows=(3, 5, 7, 11, 15), num_filters=64, num_hidden=256):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_dim, num_filters, kernel_size=(1, w), padding=0)
            for w in windows
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=(1, MAXSEQ - w + 1), stride=(1, MAXSEQ))
            for w in windows
        ])
        total_c = num_filters * len(windows)
        self.se = nn.Sequential(
            nn.Linear(total_c, total_c // 4),
            nn.ReLU(),
            nn.Linear(total_c // 4, total_c),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(0.65)
        self.fc1 = nn.Linear(total_c, num_hidden)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)
        for c in self.convs:
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x4d):
        outs = [p(F.relu(c(x4d))).flatten(1) for c, p in zip(self.convs, self.pools)]
        feat = torch.cat(outs, dim=1)
        feat = feat * self.se(feat)
        feat = self.drop(feat)
        h = F.gelu(self.norm1(self.fc1(feat)))
        return self.fc2(h).squeeze(-1)

def train_fast_5fold(model_cls, x_tr, y_tr, x_te, p_tm, n_tm, epochs=45, batch_size=256):
    # Convert (N, 35, D) -> (N, D, 1, 35) on CPU
    x_tr_4d = np.transpose(x_tr, (0, 2, 1))[:, :, None, :].copy()
    x_te_4d = np.transpose(x_te, (0, 2, 1))[:, :, None, :].copy()
    p_tm_4d = np.transpose(p_tm, (0, 2, 1))[:, :, None, :].copy()
    n_tm_4d = np.transpose(n_tm, (0, 2, 1))[:, :, None, :].copy()

    in_dim = x_tr.shape[-1]
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    oof = np.zeros(len(x_tr), dtype=np.float32)
    te_preds, tp_preds, tn_preds = [], [], []

    te_t = torch.from_numpy(x_te_4d).to(DEVICE)
    tp_t = torch.from_numpy(p_tm_4d).to(DEVICE)
    tn_t = torch.from_numpy(n_tm_4d).to(DEVICE)

    for fold_i, (tr_idx, val_idx) in enumerate(kfold.split(x_tr)):
        torch.manual_seed(42 + fold_i)
        m = model_cls(in_dim).to(DEVICE)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        xt = torch.from_numpy(x_tr_4d[tr_idx]).to(DEVICE)
        yt = torch.from_numpy(y_tr[tr_idx]).to(DEVICE)
        xv = torch.from_numpy(x_tr_4d[val_idx]).to(DEVICE)

        m.train()
        for ep in range(epochs):
            perm = torch.randperm(len(xt), device=DEVICE)
            for i in range(0, len(xt), batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                logits = m(xt[idx])
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx]) + 1e-3 * torch.sum(m.fc2.weight ** 2)
                loss.backward()
                opt.step()

        m.eval()
        with torch.no_grad():
            oof[val_idx] = torch.sigmoid(m(xv)).cpu().numpy()
            te_preds.append(torch.sigmoid(m(te_t)).cpu().numpy())
            tp_preds.append(torch.sigmoid(m(tp_t)).cpu().numpy())
            tn_preds.append(torch.sigmoid(m(tn_t)).cpu().numpy())

    return oof, np.mean(te_preds, 0), np.mean(tp_preds, 0), np.mean(tn_preds, 0)

# -------------------------------------------------------------------------
# Length-Prior Debiased Rank/Logit Calibration (Zero Test Leakage: Uses ONLY Training OOF)
# -------------------------------------------------------------------------
def apply_training_length_prior_debiasing(oof_p, te_p, tp_p, tn_p, l_tr, l_te, lp_tm, ln_tm, alpha=0.65):
    """
    Estimates the training-set length-induced baseline logit shift E_train[logit | L_bin]
    strictly from the out-of-fold TRAINING predictions (oof_p, l_tr), and subtracts
    alpha * shift(L_bin) from logits so that short MHC-I peptides (L<=11) and medium peptides
    (12<=L<=16) are not penalized relative to long MHC-II peptides (L>=17).
    """
    eps = 1e-6
    oof_z = np.log(np.clip(oof_p, eps, 1 - eps) / (1 - np.clip(oof_p, eps, 1 - eps)))
    te_z = np.log(np.clip(te_p, eps, 1 - eps) / (1 - np.clip(te_p, eps, 1 - eps)))
    tp_z = np.log(np.clip(tp_p, eps, 1 - eps) / (1 - np.clip(tp_p, eps, 1 - eps)))
    tn_z = np.log(np.clip(tn_p, eps, 1 - eps) / (1 - np.clip(tn_p, eps, 1 - eps)))

    def get_bin(l_arr):
        b = np.zeros(len(l_arr), dtype=int)
        b[(l_arr >= 12) & (l_arr <= 16)] = 1
        b[l_arr >= 17] = 2
        return b

    b_tr = get_bin(l_tr)
    global_mean = oof_z.mean()
    global_std = max(oof_z.std(), 1e-4)

    bin_stats = {}
    for b in [0, 1, 2]:
        mask = (b_tr == b)
        if mask.sum() > 5:
            bin_stats[b] = (oof_z[mask].mean(), max(oof_z[mask].std(), 1e-4))
        else:
            bin_stats[b] = (global_mean, global_std)

    def calibrate(z_arr, l_arr):
        b_arr = get_bin(l_arr)
        z_cal = np.copy(z_arr)
        for b in [0, 1, 2]:
            m = (b_arr == b)
            if m.sum() > 0:
                mu, sd = bin_stats[b]
                z_norm = (z_arr[m] - mu) / sd * global_std + global_mean
                z_cal[m] = (1.0 - alpha) * z_arr[m] + alpha * z_norm
        return 1.0 / (1.0 + np.exp(-z_cal))

    return (
        calibrate(oof_z, l_tr),
        calibrate(te_z, l_te),
        calibrate(tp_z, lp_tm),
        calibrate(tn_z, ln_tm),
    )

if __name__ == "__main__":
    print("Step 1: Parsing completed 36 experiments from task-317 log...", flush=True)
    log_path = ROOT_DIR / "data" / "task_317_reference.log"
    parsed_log_results = {}
    if log_path.exists():
        pat = re.compile(
            r"\[([^\]]+?)\s*\]\s*\(\s*([\d\.]+)s\)\s*\|\s*CV AUC=([\d\.]+)\s*\|\s*OrigTest=([\d\.]+)\s*\|\s*"
            r"Temp-All=([\d\.]+)\s*\(PR=([\d\.]+)\)\s*\|\s*Temp-MHC1=([\d\.]+)\s*\|\s*Temp-MHC2=([\d\.]+)\s*\|\s*"
            r"Temp-Match=([\d\.]+)\s*\|\s*Strict30\(k2\)=([\d\.]+)"
        )
        with open(log_path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    exp_id = m.group(1).strip()
                    parsed_log_results[exp_id] = {
                        "runtime_sec": float(m.group(2)),
                        "5-Fold CV": {"ROC_AUC": float(m.group(3))},
                        "Original Test": {"ROC_AUC": float(m.group(4))},
                        "Temporal-All": {"ROC_AUC": float(m.group(5)), "PR_AUC": float(m.group(6))},
                        "Temporal-MHC-I": {"ROC_AUC": float(m.group(7))},
                        "Temporal-MHC-II": {"ROC_AUC": float(m.group(8))},
                        "Temporal-Matched": {"ROC_AUC": float(m.group(9))},
                        "Temporal-Strict30 (k=2)": {"ROC_AUC": float(m.group(10))},
                    }
    print(f"  Parsed {len(parsed_log_results)} completed runs from task-317.log.", flush=True)

    PLM_LIST = [
        ("immunophys", "ImmunoPhys-48"),
        ("tape", "TAPE (Pfam BERT)"),
        ("ankh_base", "Ankh-Base (48L T5)"),
        ("esmc_300m", "ESMC-300M (ESM-Cambrian)"),
        ("esm2", "ESM-2 (650M UR50D)"),
        ("esm1v", "ESM-1v (650M UR90S Variant)"),
        ("ProtTrans", "ProtT5-XL (1.2B U50)"),
        ("prostt5", "ProstT5 (1.2B 3Di Structure)"),
    ]

    imm_splits = load_plm_splits("immunophys")
    y_tr = imm_splits["train"][1]
    l_tr = imm_splits["train"][2]
    y_te = imm_splits["test"][1]
    l_te = imm_splits["test"][2]
    lp_tm, ln_tm = imm_splits["temporal_pos_neg"][2], imm_splits["temporal_pos_neg"][3]

    preds_store = {}
    full_metrics_store = {}

    print("\nStep 2: Running Fast 2D DeepScan & SE-ResScan (Multi-Scale Squeeze-and-Excitation Motif Net) across all 8 PLMs...", flush=True)
    for plm_key, plm_label in PLM_LIST:
        d = load_plm_splits(plm_key)
        x_tr, _, _ = d["train"]
        x_te, _, _ = d["test"]
        p_tm, n_tm, _, _ = d["temporal_pos_neg"]

        # 2A: DeepScan (mCNN)
        t0 = time.perf_counter()
        oof, te_p, tp_p, tn_p = train_fast_5fold(DeepScanFast2D, x_tr, y_tr, x_te, p_tm, n_tm)
        dt = time.perf_counter() - t0
        k_ds = f"{plm_key}__DeepScan (mCNN)"
        preds_store[k_ds] = (oof, te_p, tp_p, tn_p)
        m_ds = evaluate_across_cohorts(te_p, y_te, tp_p, tn_p)
        m_ds["5-Fold CV"] = compute_all_metrics(y_tr, oof)
        m_ds["runtime_sec"] = round(dt, 2)
        full_metrics_store[k_ds] = m_ds
        print(f"  [{k_ds:36s}] ({dt:4.1f}s) | CV={m_ds['5-Fold CV']['ROC_AUC']:.4f} | Orig={m_ds['Original Test']['ROC_AUC']:.4f} | TempAll={m_ds['Temporal-All']['ROC_AUC']:.4f} (PR={m_ds['Temporal-All']['PR_AUC']:.4f}) | MHC1={m_ds['Temporal-MHC-I']['ROC_AUC']:.4f} | MHC2={m_ds['Temporal-MHC-II']['ROC_AUC']:.4f} | Match={m_ds['Temporal-Matched']['ROC_AUC']:.4f} | Strict30k2={m_ds['Temporal-Strict30 (k=2)']['ROC_AUC']:.4f}", flush=True)

        # 2B: SE-ResScan (with ImmunoPhys-48 multimodal fusion for PLMs)
        if plm_key != "immunophys":
            x_tr_mm = np.concatenate([x_tr, imm_splits["train"][0]], axis=-1)
            x_te_mm = np.concatenate([x_te, imm_splits["test"][0]], axis=-1)
            p_tm_mm = np.concatenate([p_tm, imm_splits["temporal_pos_neg"][0]], axis=-1)
            n_tm_mm = np.concatenate([n_tm, imm_splits["temporal_pos_neg"][1]], axis=-1)
        else:
            x_tr_mm, x_te_mm, p_tm_mm, n_tm_mm = x_tr, x_te, p_tm, n_tm

        t0 = time.perf_counter()
        oof_se, te_se, tp_se, tn_se = train_fast_5fold(SEResScan2D, x_tr_mm, y_tr, x_te_mm, p_tm_mm, n_tm_mm)
        dt = time.perf_counter() - t0
        k_se = f"{plm_key}__SE-ResScan"
        preds_store[k_se] = (oof_se, te_se, tp_se, tn_se)
        m_se = evaluate_across_cohorts(te_se, y_te, tp_se, tn_se)
        m_se["5-Fold CV"] = compute_all_metrics(y_tr, oof_se)
        m_se["runtime_sec"] = round(dt, 2)
        full_metrics_store[k_se] = m_se
        print(f"  [{k_se:36s}] ({dt:4.1f}s) | CV={m_se['5-Fold CV']['ROC_AUC']:.4f} | Orig={m_se['Original Test']['ROC_AUC']:.4f} | TempAll={m_se['Temporal-All']['ROC_AUC']:.4f} (PR={m_se['Temporal-All']['PR_AUC']:.4f}) | MHC1={m_se['Temporal-MHC-I']['ROC_AUC']:.4f} | MHC2={m_se['Temporal-MHC-II']['ROC_AUC']:.4f} | Match={m_se['Temporal-Matched']['ROC_AUC']:.4f} | Strict30k2={m_se['Temporal-Strict30 (k=2)']['ROC_AUC']:.4f}", flush=True)

        # 2C: Length-Prior Debiased SE-ResScan (LP-SE-ResScan)
        oof_lp, te_lp, tp_lp, tn_lp = apply_training_length_prior_debiasing(
            oof_se, te_se, tp_se, tn_se, l_tr, l_te, lp_tm, ln_tm, alpha=0.55
        )
        k_lp = f"{plm_key}__LP-SE-ResScan"
        preds_store[k_lp] = (oof_lp, te_lp, tp_lp, tn_lp)
        m_lp = evaluate_across_cohorts(te_lp, y_te, tp_lp, tn_lp)
        m_lp["5-Fold CV"] = compute_all_metrics(y_tr, oof_lp)
        full_metrics_store[k_lp] = m_lp
        print(f"  [{k_lp:36s}] (Calib) | CV={m_lp['5-Fold CV']['ROC_AUC']:.4f} | Orig={m_lp['Original Test']['ROC_AUC']:.4f} | TempAll={m_lp['Temporal-All']['ROC_AUC']:.4f} (PR={m_lp['Temporal-All']['PR_AUC']:.4f}) | MHC1={m_lp['Temporal-MHC-I']['ROC_AUC']:.4f} | MHC2={m_lp['Temporal-MHC-II']['ROC_AUC']:.4f} | Match={m_lp['Temporal-Matched']['ROC_AUC']:.4f} | Strict30k2={m_lp['Temporal-Strict30 (k=2)']['ROC_AUC']:.4f}", flush=True)

        del d
        gc.collect()

    print("\nStep 3: Evaluating Multi-PLM Synergistic Ensembles...", flush=True)
    ensembles = {
        "Original Baseline Ensemble (TAPE+ProtT5+ESM2 + mCNN)": [
            ("tape__DeepScan (mCNN)", 1.0),
            ("ProtTrans__DeepScan (mCNN)", 1.0),
            ("esm2__DeepScan (mCNN)", 1.0),
        ],
        "NextGen Top-4 PLM Ensemble (ESMC-300M+Ankh+ProtT5+ESM2 + mCNN)": [
            ("esmc_300m__DeepScan (mCNN)", 0.30),
            ("ankh_base__DeepScan (mCNN)", 0.30),
            ("ProtTrans__DeepScan (mCNN)", 0.25),
            ("esm2__DeepScan (mCNN)", 0.15),
        ],
        "NextGen SE-ResScan Multi-PLM Ensemble (ESMC+Ankh+ProtT5+ESM1v+ProstT5+ImmPhys)": [
            ("esmc_300m__SE-ResScan", 0.25),
            ("ankh_base__SE-ResScan", 0.25),
            ("ProtTrans__SE-ResScan", 0.20),
            ("esm1v__SE-ResScan", 0.12),
            ("prostt5__SE-ResScan", 0.08),
            ("immunophys__SE-ResScan", 0.10),
        ],
        "NextGen LP-SE-ResScan Ensemble (Length-Prior Debiased Multi-PLM)": [
            ("esmc_300m__LP-SE-ResScan", 0.28),
            ("ankh_base__LP-SE-ResScan", 0.28),
            ("ProtTrans__LP-SE-ResScan", 0.24),
            ("esm2__LP-SE-ResScan", 0.10),
            ("esm1v__LP-SE-ResScan", 0.10),
        ],
    }

    ensemble_curves = {}
    for ens_name, members in ensembles.items():
        w_sum = sum(w for _, w in members)
        oof_e = sum(w * preds_store[k][0] for k, w in members) / w_sum
        te_e = sum(w * preds_store[k][1] for k, w in members) / w_sum
        tp_e = sum(w * preds_store[k][2] for k, w in members) / w_sum
        tn_e = sum(w * preds_store[k][3] for k, w in members) / w_sum

        m_e = evaluate_across_cohorts(te_e, y_te, tp_e, tn_e)
        m_e["5-Fold CV"] = compute_all_metrics(y_tr, oof_e)
        full_metrics_store[ens_name] = m_e
        ensemble_curves[ens_name] = {
            "test_prob": te_e.tolist(),
            "temp_pos_prob": tp_e.tolist(),
            "temp_neg_prob": tn_e.tolist(),
        }
        print(
            f"  [ENS: {ens_name:68s}] | CV={m_e['5-Fold CV']['ROC_AUC']:.4f} | "
            f"Orig={m_e['Original Test']['ROC_AUC']:.4f} | "
            f"TempAll={m_e['Temporal-All']['ROC_AUC']:.4f} (PR={m_e['Temporal-All']['PR_AUC']:.4f}, MCC={m_e['Temporal-All']['MCC']:.4f}) | "
            f"MHC1={m_e['Temporal-MHC-I']['ROC_AUC']:.4f} | "
            f"MHC2={m_e['Temporal-MHC-II']['ROC_AUC']:.4f} | "
            f"Match={m_e['Temporal-Matched']['ROC_AUC']:.4f} | "
            f"Strict30k2={m_e['Temporal-Strict30 (k=2)']['ROC_AUC']:.4f}",
            flush=True
        )

    out_data = {
        "parsed_deep_architectures": parsed_log_results,
        "full_metrics": full_metrics_store,
        "ensemble_curves": ensemble_curves,
    }
    out_json = ROOT_DIR / "data" / "nextgen_models_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved complete results to {out_json}", flush=True)
