from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
import os
import json
import shutil
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(str(ROOT_DIR))

in_json = ROOT_DIR / "data" / "nextgen_models_results.json"
with open(in_json, "r", encoding="utf-8") as f:
    res_data = json.load(f)

parsed = res_data["parsed_deep_architectures"]
full_m = res_data["full_metrics"]
curves = res_data["ensemble_curves"]

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "Arial Unicode MS", "PingFang TC", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=220)
fig.patch.set_facecolor("#ffffff")

# -------------------------------------------------------------------------
# Panel A: 8 PLM Backbone Comparison across Key Cohorts (DeepScan Baseline)
# -------------------------------------------------------------------------
ax = axes[0, 0]
plm_keys = ["tape", "esm2", "prostt5", "immunophys", "esm1v", "ProtTrans", "ankh_base", "esmc_300m"]
plm_names = [
    "TAPE\n(Pfam 38M)",
    "ESM-2\n(650M U50)",
    "ProstT5\n(3Di 1.2B)",
    "ImmunoPhys\n(48D Bio)",
    "ESM-1v\n(Variant U90)",
    "ProtT5-XL\n(1.2B U50)",
    "Ankh-Base\n(48L T5)",
    "ESMC-300M\n(Cambrian)",
]

x = np.arange(len(plm_keys))
w = 0.25

def get_auc(exp_key, cohort):
    if exp_key in parsed and cohort in parsed[exp_key]:
        return parsed[exp_key][cohort]["ROC_AUC"]
    if exp_key in full_m and cohort in full_m[exp_key]:
        return full_m[exp_key][cohort]["ROC_AUC"]
    return 0.0

auc_temp_all = [get_auc(f"{k}__DeepScan (mCNN)", "Temporal-All") for k in plm_keys]
auc_temp_mhc2 = [get_auc(f"{k}__DeepScan (mCNN)", "Temporal-MHC-II") for k in plm_keys]
auc_temp_match = [get_auc(f"{k}__DeepScan (mCNN)", "Temporal-Matched") for k in plm_keys]

b1 = ax.bar(x - w, auc_temp_all, width=w, label="Temporal-All (n=1,409)", color="#2563eb", alpha=0.9)
b2 = ax.bar(x, auc_temp_mhc2, width=w, label="Temporal-MHC-II (n=116)", color="#10b981", alpha=0.9)
b3 = ax.bar(x + w, auc_temp_match, width=w, label="Temporal-Matched (n=1,036)", color="#f59e0b", alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(plm_names, fontsize=8.5, fontweight="bold")
ax.set_ylim(0.55, 0.98)
ax.set_ylabel("ROC-AUC", fontsize=10, fontweight="bold")
ax.set_title("A. 8 種蛋白質語言模型 (PLM) 與生物物理表徵於時間序測試集之表現", fontsize=11.5, fontweight="bold", pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper left", fontsize=8.5, frameon=True)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)

# -------------------------------------------------------------------------
# Panel B: 6 Deep Learning Architectures Comparison across Cohorts (ProtT5 / Multi-PLM)
# -------------------------------------------------------------------------
ax = axes[0, 1]
arch_labels = [
    "DeepScan\n(Original mCNN)",
    "BiGRU-\nCrossAttn",
    "Tramba-\nImmunoNet",
    "Masked-\nMSCNN",
    "SE-ResScan\n(Multi-Scale SE)",
    "NextGen Multi-PLM\nSynergistic Ens.",
]
arch_keys = [
    "ProtTrans__DeepScan (mCNN)",
    "ProtTrans__BiGRU-CrossAttn",
    "ProtTrans__Tramba-ImmunoNet",
    "ProtTrans__Masked-MSCNN",
    "ProtTrans__SE-ResScan",
    "NextGen SE-ResScan Multi-PLM Ensemble (ESMC+Ankh+ProtT5+ESM1v+ProstT5+ImmPhys)",
]

x2 = np.arange(len(arch_labels))
w2 = 0.20
c_orig = [get_auc(k, "Original Test") for k in arch_keys]
c_match = [get_auc(k, "Temporal-Matched") for k in arch_keys]
c_mhc2 = [get_auc(k, "Temporal-MHC-II") for k in arch_keys]
c_strict = [get_auc(k, "Temporal-Strict30 (k=2)") for k in arch_keys]

ax.bar(x2 - 1.5 * w2, c_orig, width=w2, label="Original Test (n=1,404)", color="#6366f1")
ax.bar(x2 - 0.5 * w2, c_match, width=w2, label="Temporal-Matched (n=1,036)", color="#f59e0b")
ax.bar(x2 + 0.5 * w2, c_mhc2, width=w2, label="Temporal-MHC-II (n=116)", color="#10b981")
ax.bar(x2 + 1.5 * w2, c_strict, width=w2, label="Temporal-Strict30 k=2 (n=505)", color="#ec4899")

ax.set_xticks(x2)
ax.set_xticklabels(arch_labels, fontsize=8.5, fontweight="bold")
ax.set_ylim(0.60, 0.98)
ax.set_ylabel("ROC-AUC", fontsize=10, fontweight="bold")
ax.set_title("B. 6 種深度學習架構於獨立測試集與嚴格去同源時間序子集之比較", fontsize=11.5, fontweight="bold", pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="lower right", fontsize=8.2, frameon=True)

# -------------------------------------------------------------------------
# Panel C: ROC Curves on Temporal-All (Baseline vs NextGen Ensembles)
# -------------------------------------------------------------------------
ax = axes[1, 0]
y_temp = None
for ens_name, cdata in curves.items():
    tp = np.array(cdata["temp_pos_prob"])
    tn = np.array(cdata["temp_neg_prob"])
    y_temp = np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])
    p_temp = np.concatenate([tp, tn])
    fpr, tpr, _ = roc_curve(y_temp, p_temp)
    auc_v = metrics.auc(fpr, tpr)
    short_lbl = ens_name.split(" (")[0]
    ax.plot(fpr, tpr, lw=2.2, label=f"{short_lbl} (AUC = {auc_v:.4f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.5000)")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10, fontweight="bold")
ax.set_title("C. Temporal-All (2024–2026 全部新增 1,409 筆) ROC 曲線比較", fontsize=11.5, fontweight="bold", pad=10)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="lower right", fontsize=8.5, frameon=True)

# -------------------------------------------------------------------------
# Panel D: Best Model Performance Summary Across All 7 Cohorts (Original Baseline vs NextGen Best)
# -------------------------------------------------------------------------
ax = axes[1, 1]
cohorts = [
    "Original Test",
    "Temporal-Matched",
    "Temporal-MHC-II",
    "Temporal-All",
    "Temporal-MHC-I",
    "Temporal-Strict30 (k=2)",
]
cohort_short = [
    "Original Test\n(n=1,404)",
    "Temporal-Matched\n(n=1,036)",
    "Temporal-MHC-II\n(n=116)",
    "Temporal-All\n(n=1,409)",
    "Temporal-MHC-I\n(n=1,207)",
    "Strict30 (k=2)\n(n=505)",
]

base_aucs = [get_auc("ProtTrans__DeepScan (mCNN)", c) for c in cohorts]

best_aucs = []
for c in cohorts:
    all_vals = []
    for d_dict in [parsed, full_m]:
        for k, v in d_dict.items():
            if c in v and "ROC_AUC" in v[c]:
                all_vals.append(v[c]["ROC_AUC"])
    best_aucs.append(max(all_vals) if all_vals else 0.0)

x4 = np.arange(len(cohorts))
w4 = 0.34
b_base = ax.bar(x4 - w4 / 2, base_aucs, width=w4, label="Original Baseline (ProtTrans + DeepScan mCNN)", color="#94a3b8")
b_best = ax.bar(x4 + w4 / 2, best_aucs, width=w4, label="Next-Gen Best Model / Multi-PLM Architecture", color="#059669")

for bar1, bar2 in zip(b_base, b_best):
    h1, h2 = bar1.get_height(), bar2.get_height()
    diff = h2 - h1
    ax.text(bar1.get_x() + bar1.get_width() / 2, h1 + 0.005, f"{h1:.4f}", ha="center", va="bottom", fontsize=8)
    ax.text(bar2.get_x() + bar2.get_width() / 2, h2 + 0.005, f"{h2:.4f}\n(+{diff:.4f})", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#047857")

ax.set_xticks(x4)
ax.set_xticklabels(cohort_short, fontsize=8.5, fontweight="bold")
ax.set_ylim(0.58, 1.00)
ax.set_ylabel("ROC-AUC", fontsize=10, fontweight="bold")
ax.set_title("D. 跨全數測試子集之效能突破：原版基準 vs. 新世代最佳模型", fontsize=11.5, fontweight="bold", pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=8.8, frameon=True)

plt.tight_layout()
out_dir = ROOT_DIR / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "nextgen_plm_dl_results_en.png"
plt.savefig(str(out_png), dpi=220)
print(f"Saved plot to {out_png}", flush=True)
