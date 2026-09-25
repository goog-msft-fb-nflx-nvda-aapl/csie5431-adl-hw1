"""Generate static report figures from results_summary.csv.
Palette: validated default (docs/survey dataviz skill reference).
Light-surface, print-oriented PNGs (300 dpi) for embedding in report.pdf.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

# ---- palette (references/palette.md, light mode) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dc"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": False,
    "font.size": 11,
    "font.family": "DejaVu Sans",
})

OUT = "/private/tmp/claude-501/-Users-chun-feitan-Desktop-CSIE5431/301d11e8-fb04-4fa0-ba7c-fb59f29c5b9f/scratchpad/report_package/figures"

df = pd.read_csv("/private/tmp/results_summary.csv")


def hbar(ax, labels, values, colors, xlabel, xlim=None, highlight_label=None):
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(left=False)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    for yi, v in zip(y, values):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01 if xlim else v + 0.005, yi, f"{v:.3f}",
                 va="center", ha="left", fontsize=9.5, color=TEXT_PRIMARY)
    return bars


# ---------------------------------------------------------------
# Fig 1: epoch-budget correction (bilingual mBERT / XLM-R)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.2))
rows = [
    ("mBERT, 6 epochs", "A1_mbert_lastk2_bce", BLUE),
    ("mBERT, 20 epochs", "E1_mbert_lastk2_bce_ep20", BLUE),
    ("XLM-R, 6 epochs", "A2_xlmr_lastk2_bce", ORANGE),
    ("XLM-R, 10 epochs", "A2b_xlmr_lastk2_bce_ep10", ORANGE),
]
labels = [r[0] for r in rows]
vals = [df[df.run_name == r[1]].dev_tuned_macro.iloc[0] for r in rows]
colors = [r[2] for r in rows]
hbar(ax, labels, vals, colors, "Dev macro-F1 (tuned thresholds)", xlim=(0, 0.85))
ax.set_title("Early bilingual baselines were undertrained\n(short epoch budgets used before the epoch-count issue was found)",
             fontsize=11, color=TEXT_PRIMARY, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_epoch_budget_correction.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Fig 2: zh-route model comparison (best config per architecture)
# ---------------------------------------------------------------
zh_rows = [
    ("MacBERT (zh-specific)", "C2b_macbert_none_wbce_zh_ep20"),
    ("mDeBERTa-v3", "F1_mdeberta_zh"),
    ("Qwen2.5-0.5B", "G1_qwen05_zh"),
    ("BGE-large-zh-v1.5", "P2_bgelarge_zh"),
    ("multilingual-e5-large", "P4_me5large_zh"),
    ("BGE-M3 (DB-Loss)", "O1_bgem3_zh_dbloss"),
    ("BGE-M3 (weighted BCE) — submitted", "F3_bgem3_zh"),
]
labels = [r[0] for r in zh_rows]
vals = [df[df.run_name == r[1]].dev_tuned_macro.iloc[0] for r in zh_rows]
colors = [ORANGE if "submitted" in l else BLUE for l in labels]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
hbar(ax, labels, vals, colors, "Dev macro-F1 (tuned thresholds)", xlim=(0, 1.0))
ax.set_title("Chinese route: best-checkpoint comparison across backbones", fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_zh_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Fig 3: en-route model comparison (best config per architecture)
# ---------------------------------------------------------------
en_rows = [
    ("SpanBERT", "A5_spanbert_lastk2_bce_en"),
    ("DeBERTa-v3", "A4b_deberta_lastk2_bce_en"),
    ("RoBERTa-base", "C3b_roberta_none_asl_en_ep20"),
    ("RoBERTa + MultiWOZ DAPT", "H2_roberta_dapt_en"),
    ("Qwen2.5-0.5B", "G2_qwen05_en"),
    ("mDeBERTa-v3", "F2_mdeberta_en"),
    ("multilingual-e5-large", "P3_me5large_en"),
    ("BGE-large-en-v1.5", "P1_bgelarge_en"),
    ("BGE-M3 (plain ASL)", "F4_bgem3_en"),
    ("BGE-M3 + zh warm-start + DB-Loss — submitted", "N1_bgem3_en_seqft_zh"),
]
labels = [r[0] for r in en_rows]
vals = [df[df.run_name == r[1]].dev_tuned_macro.iloc[0] for r in en_rows]
colors = [ORANGE if "submitted" in l else BLUE for l in labels]
fig, ax = plt.subplots(figsize=(9, 4.4))
hbar(ax, labels, vals, colors, "Dev macro-F1 (tuned thresholds)", xlim=(0, 1.0))
ax.set_title("English route: best-checkpoint comparison across backbones", fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_en_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Fig 4: context-inclusion strategy ablation
# ---------------------------------------------------------------
zh_ctx = [("None", "C2b_macbert_none_wbce_zh_ep20"),
          ("Last-2 turns", "A3_macbert_lastk2_bce_zh"),
          ("Last-4 turns", "B7_macbert_lastk4_zh_ep20"),
          ("Head+tail", "B8_macbert_headtail_zh_ep20")]
en_ctx = [("None", "C3b_roberta_none_asl_en_ep20"),
          ("Last-2 turns", "A4_roberta_lastk2_bce_en"),
          ("Last-4 turns", "B3_roberta_lastk4_bce_en"),
          ("Head+tail", "B5_roberta_headtail_bce_en")]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for ax, rows, title, color in [(axes[0], zh_ctx, "Chinese (MacBERT)", BLUE),
                                (axes[1], en_ctx, "English (RoBERTa)", AQUA)]:
    labels = [r[0] for r in rows]
    vals = [df[df.run_name == r[1]].dev_tuned_macro.iloc[0] for r in rows]
    x = np.arange(len(labels))
    colors = [ORANGE if l == "None" else color for l in labels]
    bars = ax.bar(x, vals, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_title(title, fontsize=10.5, loc="left")
axes[0].set_ylabel("Dev macro-F1 (tuned thresholds)")
fig.suptitle("Dropping dialogue context entirely beats every context-inclusion strategy tried",
             fontsize=11.5, x=0.02, ha="left", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_context_ablation.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Fig 5: per-class delta F1 from dropping context (none minus context)
# ---------------------------------------------------------------
def per_class(run_name):
    row = df[df.run_name == run_name].iloc[0]
    return json.loads(row["dev_per_class_f1"])

zh_none = per_class("C2b_macbert_none_wbce_zh_ep20")
zh_ctx_ = per_class("B8_macbert_headtail_zh_ep20")
en_none = per_class("B1_roberta_none_bce_en")
en_ctx_ = per_class("A4_roberta_lastk2_bce_en")

classes = list(zh_none.keys())
zh_delta = [zh_none[c] - zh_ctx_[c] for c in classes]
en_delta = [en_none[c] - en_ctx_[c] for c in classes]

order = np.argsort(en_delta)[::-1]
classes_o = [classes[i] for i in order]
zh_delta_o = [zh_delta[i] for i in order]
en_delta_o = [en_delta[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 4.5))
y = np.arange(len(classes_o))
h = 0.32
ax.barh(y + h / 2, zh_delta_o, height=h, color=BLUE, label="Chinese (none − head+tail)")
ax.barh(y - h / 2, en_delta_o, height=h, color=AQUA, label="English (none − last-2 turns)")
ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(classes_o)
ax.invert_yaxis()
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlabel("Δ per-class dev F1 (dropping context − keeping context)")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2, fontsize=9.5)
ax.set_title("Per-class effect of dropping dialogue context (positive = context hurts)",
             fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_per_class_context_delta.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Fig 6: Chinese vs English / cross-lingual transfer
# ---------------------------------------------------------------
cl_rows = [
    ("mBERT\n(bilingual, all data)", "E1_mbert_lastk2_bce_ep20"),
    ("MacBERT\n(zh-only)", "A3_macbert_lastk2_bce_zh"),
    ("RoBERTa\n(en-only)", "C3b_roberta_none_asl_en_ep20"),
    ("BGE-M3\n(zh-only)", "F3_bgem3_zh"),
    ("BGE-M3\n(en-only)", "F4_bgem3_en"),
]
labels = [r[0] for r in cl_rows]
zh_scores = [df[df.run_name == r[1]].public_zh_macro.iloc[0] for r in cl_rows]
en_scores = [df[df.run_name == r[1]].public_en_macro.iloc[0] for r in cl_rows]

x = np.arange(len(labels))
w = 0.32
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - w / 2, zh_scores, width=w, color=BLUE, label="Tested on Chinese examples")
ax.bar(x + w / 2, en_scores, width=w, color=ORANGE, label="Tested on English examples")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylim(0, 1.0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.set_ylabel("Public-test macro-F1")
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
for xi, (zv, ev) in enumerate(zip(zh_scores, en_scores)):
    ax.text(xi - w / 2, zv + 0.02, f"{zv:.2f}", ha="center", fontsize=8.5, color=TEXT_PRIMARY)
    ax.text(xi + w / 2, ev + 0.02, f"{ev:.2f}", ha="center", fontsize=8.5, color=TEXT_PRIMARY)
ax.set_title("Same-language specialists collapse off-language — except BGE-M3, which transfers cross-lingually",
             fontsize=10.8, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_cross_lingual_transfer.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# Fig 7: session progression / milestone chart
# ---------------------------------------------------------------
milestones = [
    ("Candidate A\n(single ckpt/route,\ndev-selected)", 0.8057, 0.8238),
    ("Candidate B\n(ensembled +\nLR-tuned)", 0.8218, 0.8316),
    ("Candidate C\n(BGE-M3,\nsingle ckpt/route)", 0.8264, 0.8374),
    ("Final submitted\n(zh single +\nen 2-way ensemble)", 0.8396, 0.8538),
]
labels = [m[0] for m in milestones]
macro = [m[1] for m in milestones]
micro = [m[2] for m in milestones]
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8.8, 4.4))
ax.plot(x, macro, color=BLUE, marker="o", markersize=8, linewidth=2, label="Macro-F1")
ax.plot(x, micro, color=AQUA, marker="o", markersize=8, linewidth=2, label="Micro-F1")
ax.axhline(0.82, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
ax.axhline(0.83, color=GRID, linewidth=1, linestyle=(0, (4, 3)))
ax.text(-0.35, 0.823, "Challenge tier — macro 0.82 / micro 0.83", fontsize=8.5,
        color=TEXT_SECONDARY, va="bottom", ha="left")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.75, 0.90)
ax.set_xlim(-0.4, 3.85)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.set_ylabel("Public-test score")
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
for xi, (mv, miv) in enumerate(zip(macro, micro)):
    ax.text(xi, mv - 0.018, f"{mv:.4f}", ha="center", fontsize=8.5, color=TEXT_PRIMARY)
    ax.text(xi, miv + 0.008, f"{miv:.4f}", ha="center", fontsize=8.5, color=TEXT_PRIMARY)
ax.set_title("Candidate progression: public-test score across the session", fontsize=11, loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_progression.png", dpi=300, bbox_inches="tight")
plt.close()

print("done")
