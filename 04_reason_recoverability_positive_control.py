# =====================================================================
# NB4 (single cell): Are stated reasons recoverable from the features?
#                    + Positive control for the validation protocol
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
#
# Why: NB3 found explanations do NOT align with stated reasons.
#      Before claiming "XAI fails", we must separate two explanations:
#        (i)  the features carry reason signal but XAI misses it, or
#        (ii) the features carry no reason signal at all.
#   A. Univariate association: feature vs reason group (effect sizes)
#   B. Recoverability: a supervised reason classifier = upper bound
#   C. Positive control: synthetic reasons with KNOWN signal strength ->
#      does the NB3 protocol detect alignment when it truly exists?
#
# Input : NB1 + NB3 output in /kaggle/working
# Output: /kaggle/working/nb4/   RUNTIME: ~5-10 min
# =====================================================================

import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

N_PERM_CLF = 200     # permutations for the reason-classifier significance test
N_REP_CTRL = 20      # replicates per signal strength in the positive control
SIGNALS    = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
SEED       = 42

import os, json, glob, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, make_scorer
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

BASE = os.environ.get("NB_IN", WORK_DIR)
def find_file(name):
    for p in [os.path.join(BASE, name)] + glob.glob(f"{DATA_DIR}/**/{name}", recursive=True):
        if os.path.exists(p): return p
    raise FileNotFoundError(f"{name} not found. Run NB1-NB3 first.")

OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb4")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight"); plt.show()

cfg  = json.load(open(find_file("config.json")))
data = pd.read_csv(find_file("telco_clean.csv"))
NUM = [f for f in cfg["num_features"] if f != "Avg Monthly Spend"]
BIN, CAT = cfg["bin_features"], cfg["cat_features"]
FEATS = NUM + BIN + CAT

GROUPS = ["Price", "Support", "Competitor", "Quality"]
ch = data[(data[cfg["target"]] == 1) & data["Reason Group"].isin(GROUPS)].reset_index(drop=True)
Xc, rg = ch[FEATS], ch["Reason Group"].values
print(f"Actionable churners: {len(ch)} | groups: {pd.Series(rg).value_counts().to_dict()}")

# ---------------------------------------------------------------------
# A. Univariate association between each feature and reason group
#    numeric: Kruskal-Wallis + epsilon^2 ; categorical/binary: chi2 + Cramer's V
#    Holm correction over all features
# ---------------------------------------------------------------------
rows = []
for f in FEATS:
    if f in NUM:
        groups = [ch.loc[rg == g, f].values for g in GROUPS]
        H, p = stats.kruskal(*groups)
        eff = max(0.0, (H - len(GROUPS) + 1) / (len(ch) - len(GROUPS)))      # epsilon^2
        means = {g: ch.loc[rg == g, f].mean() for g in GROUPS}
        rows.append({"Feature": f, "Test": "Kruskal-Wallis", "Stat": H, "p": p,
                     "Effect": eff, "Effect type": "epsilon²", **{f"{g} mean/%": round(v, 2) for g, v in means.items()}})
    else:
        ct = pd.crosstab(ch[f], rg)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (len(ch) * (min(ct.shape) - 1)))
        top = ch[f].value_counts().index[0]
        pct = {g: (ch.loc[rg == g, f] == top).mean() * 100 for g in GROUPS}
        rows.append({"Feature": f + (f" [= {top}]" if f in CAT else ""), "Test": "Chi-square", "Stat": chi2, "p": p,
                     "Effect": v, "Effect type": "Cramér's V", **{f"{g} mean/%": round(x, 1) for g, x in pct.items()}})
uni = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
m = len(uni)
uni["p_holm"] = np.maximum.accumulate(np.minimum(1, uni["p"] * (m - np.arange(m))))
uni["Significant (Holm .05)"] = uni["p_holm"] < 0.05
uni.round(4).to_csv(os.path.join(TAB, "T9_univariate_feature_by_reason.csv"), index=False)
print("\n=== Table 9: Feature vs stated reason group (actionable churners) ===")
display(uni.round(4))
print(f"Features significant after Holm: {int(uni['Significant (Holm .05)'].sum())} / {m} | "
      f"max effect size = {uni['Effect'].max():.3f} (small effect: V≈0.1, eps²≈0.01)")

# ---------------------------------------------------------------------
# B. Recoverability: can ANY model predict the stated reason from features?
#    (upper bound on what an explanation could possibly reflect)
# ---------------------------------------------------------------------
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("bin", "passthrough", BIN),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
CLFS = {
    "Multinomial LogReg": LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000),
    "Gradient Boosting":  HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                         class_weight="balanced", random_state=SEED),
}
le = LabelEncoder().fit(GROUPS)
y4 = le.transform(rg)
y3 = LabelEncoder().fit_transform(pd.Series(rg).replace({"Competitor": "Offering", "Quality": "Offering"}))
macro_auc = make_scorer(roc_auc_score, response_method="predict_proba", multi_class="ovr", average="macro")

rec_rows = []
for task, yy in [("4 reason groups", y4), ("3 domains (Price/Support/Offering)", y3)]:
    for name, clf in CLFS.items():
        pipe = Pipeline([("pre", make_pre()), ("clf", clf)])
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
        aucs, baccs = [], []
        for tr, te in cv.split(Xc, yy):
            p = pipe.fit(Xc.iloc[tr], yy[tr]).predict_proba(Xc.iloc[te])
            aucs.append(roc_auc_score(yy[te], p, multi_class="ovr", average="macro"))
            baccs.append(balanced_accuracy_score(yy[te], p.argmax(1)))
        t0 = time.time()
        score, perm_scores, p_perm = permutation_test_score(
            pipe, Xc, yy, scoring=macro_auc, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            n_permutations=N_PERM_CLF, random_state=SEED, n_jobs=-1)
        rec_rows.append({"Task": task, "Model": name,
                         "Macro AUC": f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
                         "Balanced acc.": f"{np.mean(baccs):.3f} ± {np.std(baccs):.3f}",
                         "Chance bal. acc.": round(1 / len(np.unique(yy)), 3),
                         "Perm. null AUC (mean)": round(perm_scores.mean(), 3),
                         "Perm. p": round(p_perm, 4)})
        print(f"{task:36s} {name:20s} AUC {np.mean(aucs):.3f} | perm p {p_perm:.4f} | {(time.time()-t0)/60:.1f} min")
rec = pd.DataFrame(rec_rows)
rec.to_csv(os.path.join(TAB, "T10_reason_recoverability.csv"), index=False)
print("\n=== Table 10: Can stated reasons be predicted from the features at all? ==="); display(rec)

# ---------------------------------------------------------------------
# C. Positive control: inject reasons with KNOWN feature dependence,
#    then run the same SHAP-reason validation as NB3 (primary model, primary map)
# ---------------------------------------------------------------------
shap_file = None
for m_ in ["CatBoost", "XGBoost", "LogReg"]:          # primary model first
    try: shap_file = find_file(f"nb3/shap_{m_}.csv"); break
    except FileNotFoundError: pass
assert shap_file, "NB3 SHAP file not found"
model_name = os.path.basename(shap_file).replace("shap_", "").replace(".csv", "")
shap_all = pd.read_csv(shap_file)
idx = data.index[(data[cfg["target"]] == 1) & data["Reason Group"].isin(GROUPS)].values
SV = shap_all.loc[idx, FEATS].values

DOMAIN_MAP = {   # identical to NB3 primary map
    "Price":    ["Monthly Charges", "Total Charges", "Charge Gap"],
    "Support":  ["Tech Support", "Online Security", "Device Protection"],
    "Offering": ["Internet Service", "Streaming TV", "Streaming Movies", "Multiple Lines",
                 "Online Backup", "Phone Service", "Num Services"],
}
ACT = list(DOMAIN_MAP)
pos = np.clip(SV, 0, None); tot = pos.sum(1); tot[tot == 0] = 1
SHARE = np.column_stack([pos[:, [FEATS.index(f) for f in DOMAIN_MAP[d]]].sum(1) / tot for d in ACT])

def z(v): v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-9)
d_ = data.loc[idx]
# Latent "true cause" scores built from domain features (directions that raise churn risk)
LAT = np.column_stack([
    z(z(d_["Monthly Charges"]) + 0.5 * z(d_["Charge Gap"])),                                   # Price
    z((d_["Tech Support"] == "No").astype(int) + (d_["Online Security"] == "No").astype(int)),  # Support
    z((d_["Internet Service"] == "Fiber optic").astype(int) + 0.5 * z(d_["Num Services"])),     # Offering
])

def run_validation(true_idx):
    pred = SHARE.argmax(1)
    acc = (pred == true_idx).mean()
    aucs = [roc_auc_score((true_idx == j).astype(int), SHARE[:, j]) for j in range(3)]
    return acc, aucs

ctrl = []
for s in SIGNALS:
    for r in range(N_REP_CTRL):
        logits = s * LAT
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        true_idx = np.array([rng.choice(3, p=pi) for pi in p])
        acc, aucs = run_validation(true_idx)
        null = np.mean([(SHARE.argmax(1) == rng.permutation(true_idx)).mean() for _ in range(200)])
        ctrl.append({"signal": s, "rep": r, "acc": acc, "null_acc": null,
                     **{f"AUC {d}": a for d, a in zip(ACT, aucs)}})
ctrl = pd.DataFrame(ctrl)
ctrl_sum = ctrl.groupby("signal")[[f"AUC {d}" for d in ACT] + ["acc", "null_acc"]].mean()
ctrl_sum["Mean AUC"] = ctrl_sum[[f"AUC {d}" for d in ACT]].mean(1)
ctrl_sum["Acc. gain over null"] = ctrl_sum["acc"] - ctrl_sum["null_acc"]

# real data point
true_real = pd.Series(ch["Reason Group"]).replace({"Competitor": "Offering", "Quality": "Offering"}).map(
    {d: i for i, d in enumerate(ACT)}).values
acc_r, aucs_r = run_validation(true_real)
real_row = pd.Series({**{f"AUC {d}": a for d, a in zip(ACT, aucs_r)}, "Mean AUC": np.mean(aucs_r)}, name="Real IBM reasons")
ctrl_tab = pd.concat([ctrl_sum[[f"AUC {d}" for d in ACT] + ["Mean AUC", "Acc. gain over null"]],
                      real_row.to_frame().T])
ctrl_tab.round(3).to_csv(os.path.join(TAB, "T11_positive_control.csv"))
print(f"\n=== Table 11: Positive control ({model_name} SHAP, {N_REP_CTRL} replicates per signal) ===")
display(ctrl_tab.round(3))

# Figure: detection curve
fig, ax = plt.subplots(figsize=(7.5, 4.3))
cols = {"Price": "#C8553D", "Support": "#4A6FA5", "Offering": "#5B8C5A"}
for d in ACT:
    g = ctrl.groupby("signal")[f"AUC {d}"]
    ax.plot(SIGNALS, g.mean(), "o-", color=cols[d], label=f"{d} (synthetic)")
    ax.fill_between(SIGNALS, g.quantile(0.025), g.quantile(0.975), color=cols[d], alpha=0.12)
    ax.axhline(aucs_r[ACT.index(d)], color=cols[d], ls=":", lw=1.3)
ax.axhline(0.5, color="grey", lw=0.8)
ax.set_xlabel("Injected reason–feature signal strength"); ax.set_ylabel("Domain AUC of SHAP share")
ax.set_title("Positive control: protocol detects alignment when it exists\n(dotted = observed with real IBM reasons)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
savefig("F15_positive_control")

# Figure: univariate effect sizes
fig, ax = plt.subplots(figsize=(7.5, 5.5))
u = uni.sort_values("Effect")
ax.barh(u["Feature"], u["Effect"], color=np.where(u["Significant (Holm .05)"], "#C8553D", "#B0B7C3"))
ax.axvline(0.1, ls="--", color="grey", lw=0.8); ax.text(0.1, -0.9, " small (V=0.1)", fontsize=8, color="grey")
ax.set_xlabel("Effect size (Cramér's V / epsilon²)")
ax.set_title("Association between features and stated reason\n(red = significant after Holm)")
savefig("F16_univariate_effects")

print("\nSaved to", OUT, "->", sorted(os.listdir(OUT)))
print("\nHOW TO READ:")
print(" Table 10 Perm. p < .05 and AUC clearly > 0.5  -> reasons ARE (partly) recoverable -> XAI misses real signal")
print(" Table 10 AUC ~0.5, Perm. p > .05              -> features carry no reason signal -> SHAP cannot be read as 'cause'")
print(" Table 11: synthetic AUC rising with signal     -> the NB3 protocol is valid (has power)")
