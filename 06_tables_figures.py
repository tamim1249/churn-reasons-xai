# =====================================================================
# NB6 (single cell): FINAL manuscript tables + figures
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
#
# Reads NB1-NB5 outputs from /kaggle/working and produces:
#   final/tables/   Table_1..8 (main) + Table_A1..A9 (appendix) as CSV
#   final/manuscript_tables.xlsx   (one sheet per table, caption in row 1)
#   final/figures/  Fig_2..8 as PNG (600 dpi) + PDF (vector)
#   final/captions.md  draft captions with the actual numbers
# Also runs the ROBUSTNESS check (Total Charges moved to Relationship domain)
# RUNTIME: ~5-8 min.  Fig 1 (framework diagram) must be drawn by hand (draw.io).
# =====================================================================
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

N_BOOT, N_PERM, N_REP_CTRL, SEED = 1000, 2000, 20, 42
SIGNALS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

import os, json, glob, re, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.isotonic import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

BASE = os.environ.get("NB_IN", WORK_DIR)
def ff(name, required=True):
    for p in [os.path.join(BASE, name)] + glob.glob(f"{DATA_DIR}/**/{name}", recursive=True):
        if os.path.exists(p): return p
    if required: raise FileNotFoundError(f"{name} not found - run NB1-NB5 first")
    return None
OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "final")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)

# ---------- journal style ----------
C = {"blue": "#3B5B92", "red": "#C0392B", "green": "#4E8A55", "grey": "#8C939D", "dark": "#2B2B2B",
     "light": "#DCE3EE", "orange": "#D98C2B"}
MODEL_COL = {"CatBoost": C["red"], "XGBoost": C["orange"], "LightGBM": C["green"],
             "RandomForest": C["blue"], "LogReg": C["dark"], "MLP": C["grey"]}
DOM_COL = {"Price": C["red"], "Support": C["blue"], "Offering": C["green"]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 600, "pdf.fonttype": 42})
def panel(ax, letter): ax.text(-0.12, 1.06, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold")
def savefig(name):
    plt.tight_layout()
    for ext in ["png", "pdf"]: plt.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.show()

TABLES, CAPTIONS = {}, {}
def table(key, df, caption, index=True):
    TABLES[key] = (df, caption, index); CAPTIONS[key] = caption
    df.to_csv(os.path.join(TAB, f"{key}.csv"), index=index)
    print(f"\n=== {key}: {caption[:110]}{'...' if len(caption) > 110 else ''}"); display(df)
fmt = lambda m, s, d=3: f"{m:.{d}f} ± {s:.{d}f}"

# ---------- load ----------
cfg  = json.load(open(ff("config.json")))
data = pd.read_csv(ff("telco_clean.csv"))
TARGET, VALUE = cfg["target"], cfg["value_col"]
NUM = [f for f in cfg["num_features"] if f != "Avg Monthly Spend"]
BIN, CAT = cfg["bin_features"], cfg["cat_features"]
FEATS = NUM + BIN + CAT
y, CLTV, N = data[TARGET].astype(int).values, data[VALUE].astype(float).values, len(data)
splits = [s for s in json.load(open(ff("nb2/outer_splits.json"))) if s["repeat"] == 0]
fold_of = np.empty(N, int)
for s in splits: fold_of[s["test_idx"]] = s["fold"]
SHAPM = {}
for m in ["CatBoost", "XGBoost", "LogReg"]:
    p = ff(f"nb3/shap_{m}.csv", required=False)
    if p: SHAPM[m] = pd.read_csv(p)
PRIMARY = next(iter(SHAPM))
lime_p = ff(f"nb3/lime_{PRIMARY}.csv", required=False)
LIME = pd.read_csv(lime_p) if lime_p else None
print("Explained models:", list(SHAPM), "| LIME:", LIME is not None)

GROUPS4 = ["Price", "Support", "Competitor", "Quality"]
R2D = {"Price": "Price", "Support": "Support", "Competitor": "Offering", "Quality": "Offering"}
DOMS = ["Price", "Support", "Offering"]
rg = data["Reason Group"].astype(object).values
act = (y == 1) & np.isin(rg, GROUPS4)
val_idx = np.where(act)[0]
true_dom = np.array([R2D.get(r) if isinstance(r, str) else None for r in rg], dtype=object)

MAPS = {
 "Primary": {"Price": ["Monthly Charges", "Total Charges", "Charge Gap"],
             "Support": ["Tech Support", "Online Security", "Device Protection"],
             "Offering": ["Internet Service", "Streaming TV", "Streaming Movies", "Multiple Lines", "Online Backup",
                          "Phone Service", "Num Services"],
             "Relationship": ["Contract", "Tenure Months", "Paperless Billing", "Payment Method"],
             "Demographic": ["Male", "Senior Citizen", "Partner", "Dependents", "Latitude", "Longitude"]},
}
MAPS["Alternative"] = {"Price": MAPS["Primary"]["Price"] + ["Internet Service"], "Support": ["Tech Support"],
                       "Offering": ["Online Security", "Device Protection", "Streaming TV", "Streaming Movies",
                                    "Multiple Lines", "Online Backup", "Phone Service", "Num Services"],
                       "Relationship": MAPS["Primary"]["Relationship"], "Demographic": MAPS["Primary"]["Demographic"]}
MAPS["Robustness"] = {**MAPS["Primary"], "Price": ["Monthly Charges", "Charge Gap"],
                      "Relationship": MAPS["Primary"]["Relationship"] + ["Total Charges"]}   # Total Charges ~ tenure proxy
for mp in MAPS.values(): assert sorted(f for v in mp.values() for f in v) == sorted(FEATS)

def shares(att, mp):
    pos = np.clip(att, 0, None); tot = pos.sum(1, keepdims=True); tot[tot == 0] = 1
    return pd.DataFrame({d: pos[:, [FEATS.index(f) for f in fs]].sum(1) / tot[:, 0] for d, fs in mp.items()})

# =====================================================================
# TABLE 1  Descriptives      TABLE 2  Leakage audit      TABLE 3  Reason taxonomy
# =====================================================================
t1 = pd.read_csv(ff("tables/T1_descriptives.csv"), index_col=0)
t1 = t1.drop(index=[i for i in t1.index if i.startswith("Avg Monthly Spend")], errors="ignore")
table("Table_1", t1, "Descriptive statistics by churn status (mean ± SD for continuous variables; % for binary variables).")

leak = pd.read_csv(ff("tables/T_leakage_audit.csv"), index_col=0)
leak["Decision"] = ["Excluded (target copy)", "Excluded (vendor model score)",
                    "Excluded from features; used only as validation ground truth", "Excluded from features; used in profit layer"][:len(leak)]
table("Table_2", leak, "Leakage audit: discrimination achieved by each suspicious column on its own (standalone AUC) "
                       "and its treatment in this study.")

rg_rows = []
groups_all = data.loc[y == 1, "Reason Group"].value_counts()
for g in ["Competitor", "Quality", "Support", "Price", "Unknown", "Uncontrollable"]:
    if g not in groups_all: continue
    reasons = data.loc[(y == 1) & (data["Reason Group"] == g), "Churn Reason"].value_counts()
    rg_rows.append({"Reason group": g, "Stated reasons (n)": "; ".join(f"{r} ({n})" for r, n in reasons.items()),
                    "n": int(groups_all[g]), "% of churners": round(100 * groups_all[g] / (y == 1).sum(), 1),
                    "Explanation domain": R2D.get(g, "— (excluded from validation)"),
                    "Mean CLTV": round(data.loc[(y == 1) & (data["Reason Group"] == g), VALUE].mean(), 0)})
table("Table_3", pd.DataFrame(rg_rows).set_index("Reason group"),
      "Taxonomy of stated churn reasons, their mapping to explanation domains, and customer value.")

# =====================================================================
# TABLE 4  Predictive performance (nested CV) + stats
# =====================================================================
fm = pd.read_csv(ff("nb2/fold_metrics.csv"))
auc_mat = fm.pivot(index="Fold", columns="Model", values="AUC")
order = auc_mat.mean().sort_values(ascending=False).index.tolist()
best = order[0]; J = auc_mat.shape[0]
n_te = len(splits[0]["test_idx"]); n_tr = N - n_te
ranks = auc_mat.rank(axis=1, ascending=False).mean()
fr = stats.friedmanchisquare(*[auc_mat[c] for c in auc_mat]) if auc_mat.shape[1] >= 3 else None
praw = {}
for mdl in order[1:]:
    d = auc_mat[best] - auc_mat[mdl]; v = (1 / J + n_te / n_tr) * d.var(ddof=1)
    praw[mdl] = 2 * stats.t.sf(abs(d.mean() / np.sqrt(v)), J - 1) if v > 0 else 1.0
ps = pd.Series(praw).sort_values(); mm = len(ps)
holm = pd.Series(np.maximum.accumulate(np.minimum(1, ps.values * (mm - np.arange(mm)))), index=ps.index)
rows = []
for mdl in order:
    g = fm[fm.Model == mdl]
    rows.append({"Model": mdl, **{k: fmt(g[k].mean(), g[k].std()) for k in ["AUC", "PR_AUC", "F1", "Recall", "Brier"]},
                 "Mean rank": round(ranks[mdl], 2),
                 f"p (Holm) vs {best}": "—" if mdl == best else f"{holm[mdl]:.3f}"})
t4 = pd.DataFrame(rows).set_index("Model").rename(columns={"PR_AUC": "PR-AUC"})
fr_txt = f"Friedman χ² = {fr.statistic:.2f}, p = {fr.pvalue:.1e}" if fr else ""
table("Table_4", t4, f"Out-of-sample performance under nested {J}-fold repeated cross-validation (mean ± SD over outer folds). "
                     f"{fr_txt}. p-values: Nadeau–Bengio corrected resampled t-test vs the best model, Holm-adjusted. "
                     "Threshold-based metrics at 0.5; the MLP is not class-weighted.")

# =====================================================================
# TABLE 5  Explanation stability (global + local)
# =====================================================================
SV = {m: df[FEATS].values for m, df in SHAPM.items()}
mods = list(SV)
def topk(a, k=3): return np.argsort(-np.abs(a), 1)[:, :k]
gi = pd.DataFrame({m: np.abs(SV[m]).mean(0) for m in mods}, index=FEATS)
st_rows = []
for i, a in enumerate(mods):
    for b in mods[i + 1:]:
        ja = np.array([len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(topk(SV[a]), topk(SV[b]))])
        st_rows.append({"Comparison": f"{a} vs {b}", "Global Kendall τ": round(stats.kendalltau(gi[a], gi[b])[0], 3),
                        "Local top-3 Jaccard (mean)": round(ja.mean(), 3),
                        "% customers with identical top-3": round(100 * (ja == 1).mean(), 1),
                        "ΔAUC (out-of-fold)": round(abs(roc_auc_score(y, SHAPM[a]["prob"]) - roc_auc_score(y, SHAPM[b]["prob"])), 4)})
fold_imp = pd.DataFrame({k: np.abs(SV[PRIMARY][fold_of == k]).mean(0) for k in np.unique(fold_of)}, index=FEATS)
taus = [stats.kendalltau(fold_imp[i], fold_imp[j])[0] for i in fold_imp for j in fold_imp if i < j]
st_rows.append({"Comparison": f"{PRIMARY} across {fold_imp.shape[1]} folds", "Global Kendall τ": round(np.mean(taus), 3),
                "Local top-3 Jaccard (mean)": np.nan, "% customers with identical top-3": np.nan, "ΔAUC (out-of-fold)": np.nan})
table("Table_5", pd.DataFrame(st_rows).set_index("Comparison"),
      "Stability of SHAP explanations across models and folds. Near-identical accuracy does not imply near-identical "
      "customer-level explanations (Rashomon effect).")

# =====================================================================
# TABLE 6  Explanation–reason validation (3 maps incl. robustness)
# =====================================================================
td = true_dom[val_idx]
def boot_auc(lab, sc):
    v = []
    for _ in range(N_BOOT):
        b = rng.integers(0, len(lab), len(lab))
        if lab[b].min() != lab[b].max(): v.append(roc_auc_score(lab[b], sc[b]))
    return np.percentile(v, [2.5, 97.5])
VAL, FOREST = [], []
expl = [(f"SHAP-{m}", SV[m][val_idx]) for m in mods]
if LIME is not None:
    lime_mat = np.zeros((N, len(FEATS)))
    pos_map = {cid: i for i, cid in enumerate(data[cfg["id_col"]])}
    for _, r in LIME.iterrows(): lime_mat[pos_map[r[cfg["id_col"]]]] = r[FEATS].values
    expl.append((f"LIME-{PRIMARY}", lime_mat[val_idx]))
for mp_name, mp in MAPS.items():
    for name, att in expl:
        sh = shares(att, mp)
        pred = np.array(DOMS)[sh[DOMS].values.argmax(1)]
        acc = (pred == td).mean()
        perm = np.array([(pred == rng.permutation(td)).mean() for _ in range(N_PERM)])
        row = {"Explanation": name, "Domain map": mp_name, "Alignment acc.": round(acc, 3),
               "Permutation p": round((np.sum(perm >= acc) + 1) / (N_PERM + 1), 3),
               "Majority baseline": round(pd.Series(td).value_counts(normalize=True).iloc[0], 3),
               "Cohen κ": round(cohen_kappa_score(td, pred), 3)}
        for d in DOMS:
            lab = (td == d).astype(int); auc = roc_auc_score(lab, sh[d].values); lo, hi = boot_auc(lab, sh[d].values)
            row[f"AUC {d} [95% CI]"] = f"{auc:.3f} [{lo:.3f}, {hi:.3f}]"
            FOREST.append({"Explanation": name, "Map": mp_name, "Domain": d, "AUC": auc, "lo": lo, "hi": hi})
        row["Non-actionable share"] = round(sh[["Relationship", "Demographic"]].sum(1).mean(), 3)
        VAL.append(row)
t6 = pd.DataFrame(VAL).set_index(["Domain map", "Explanation"])
FOREST = pd.DataFrame(FOREST)
table("Table_6", t6, f"Agreement between explanations and customers' stated churn reasons (n = {len(val_idx)} churners with an "
                     f"actionable reason). Domain AUC: does the domain's share of positive attribution rank customers who "
                     f"stated that reason higher (0.5 = no information)? 95% CIs from {N_BOOT} bootstrap resamples; "
                     f"permutation p from {N_PERM} label shuffles. 'Robustness' moves Total Charges to the Relationship domain.")

# =====================================================================
# TABLE 7  Reason recoverability (incl. geography ablation)
# =====================================================================
t10 = pd.read_csv(ff("nb4/tables/T10_reason_recoverability.csv"))
t12 = pd.read_csv(ff("nb4/tables/T12_recoverability_geo_ablation.csv"))
t7a = t10.assign(**{"Feature set": "All features"})[["Task", "Feature set", "Model", "Macro AUC", "Balanced acc.", "Chance bal. acc.", "Perm. p"]]
t7b = t12[t12["Feature set"] != "All features"].assign(Task="4 reason groups")
t7 = pd.concat([t7a, t7b], ignore_index=True).fillna("—").set_index(["Task", "Feature set", "Model"])
table("Table_7", t7, "Can stated reasons be predicted from customer features at all? Supervised 'why-model' trained only on "
                     "churners (repeated stratified 5-fold CV; permutation test with 200 label shuffles). The signal "
                     "disappears without geography.")

# =====================================================================
# TABLE 8  Profit of retention strategies
# =====================================================================
t16 = pd.read_csv(ff("nb5/tables/T16_profit_strategies.csv"), index_col=0)
base_name = [s for s in t16.index if s.startswith("S2 ")][0]
base = t16.loc[base_name, "Max profit"]
t8 = pd.DataFrame({"Max profit": t16["Max profit"].round(0).astype(int),
                   "% change vs S2": ((t16["Max profit"] - base) / abs(base) * 100).round(1),
                   "Δ vs S2 [95% CI]": t16.get("Δ vs S2 [95% CI]", pd.Series(index=t16.index, dtype=str)).fillna("—"),
                   "Optimal % targeted": t16["Optimal % targeted"].round(1),
                   "Profit per customer": t16["Profit per customer"].round(1),
                   "EMP per customer": t16["EMP per customer"].round(1)})
table("Table_8", t8, "Realised retention profit on out-of-fold predictions (offer success 0.40 if the offer matches the stated "
                     "reason, 0.10 otherwise; 0 for uncontrollable churn; cost 100 per contact). Paired bootstrap CIs "
                     "(1000 resamples) for the difference to the majority-offer baseline S2.")

# =====================================================================
# APPENDIX TABLES
# =====================================================================
for key, path, cap, idx in [
    ("Table_A1", "nb2/tables/T4_smote_ablation.csv", "In-fold SMOTE versus class weighting (first CV repeat).", True),
    ("Table_A2", "nb4/tables/T9_univariate_feature_by_reason.csv", "Univariate association of each feature with the stated reason group (Holm-adjusted).", False),
    ("Table_A3", "nb3/tables/T5_global_importance.csv", "Global feature importance (share of total |SHAP|) by model.", True),
    ("Table_A4", "nb3/tables/T8_domain_share_by_reason.csv", f"Mean share (%) of positive {PRIMARY} attribution per domain, by stated reason group.", True),
    ("Table_A5", "nb5/tables/T15_calibration.csv", "Probability calibration before and after cross-fitted isotonic regression.", True),
    ("Table_A6", "nb5/tables/T17_profit_by_model.csv", "Retention profit (strategy S2) by churn model.", True),
    ("Table_A7", "nb5/tables/T18_sensitivity_SHAP-guided.csv", "Sensitivity: % profit change of SHAP-guided offers vs majority offer across offer-success and cost settings.", True),
    ("Table_A8", "nb5/tables/T19_retention_playbook.csv", "Retention playbook at the profit-optimal targeting depth.", False),
]:
    p = ff(path, required=False)
    if p: table(key, pd.read_csv(p, index_col=0 if idx else None), cap, index=idx)

# =====================================================================
# Positive control (primary + robustness map) -> Table A9 + Fig 6
# =====================================================================
d_ = data.loc[val_idx]
z = lambda v: (np.asarray(v, float) - np.mean(v)) / (np.std(v) + 1e-9)
LAT = np.column_stack([z(z(d_["Monthly Charges"]) + 0.5 * z(d_["Charge Gap"])),
                       z((d_["Tech Support"] == "No").astype(int) + (d_["Online Security"] == "No").astype(int)),
                       z((d_["Internet Service"] == "Fiber optic").astype(int) + 0.5 * z(d_["Num Services"]))])
PC = []
for mp_name in ["Primary", "Robustness"]:
    SH = shares(SV[PRIMARY][val_idx], MAPS[mp_name])[DOMS].values
    for s in SIGNALS:
        for r in range(N_REP_CTRL):
            lg = s * LAT; p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            t = (p.cumsum(1) > rng.random((len(p), 1))).argmax(1)
            PC += [{"Map": mp_name, "signal": s, "Domain": d, "AUC": roc_auc_score((t == j).astype(int), SH[:, j])}
                   for j, d in enumerate(DOMS)]
PC = pd.DataFrame(PC)
a9 = PC.groupby(["Map", "signal", "Domain"])["AUC"].mean().unstack("Domain")[DOMS].round(3)
table("Table_A9", a9, f"Positive control: mean domain AUC of {PRIMARY} SHAP shares when synthetic reasons with known "
                      f"feature dependence are injected ({N_REP_CTRL} replicates per signal level).")

# =====================================================================
# FIGURES
# =====================================================================
# ---- Fig 2: data overview ----
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
overall = y.mean() * 100
for ax, col, L in zip(axes[:2], ["Contract", "Internet Service"], "ab"):
    r = data.groupby(col)[TARGET].mean().mul(100).sort_values()
    ax.barh(r.index, r.values, color=C["blue"], height=0.6)
    for i, v in enumerate(r.values): ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.axvline(overall, ls="--", lw=0.8, color=C["dark"]); ax.set_xlim(0, r.max() * 1.25)
    ax.set_xlabel("Churn rate (%)"); ax.set_title(f"Churn by {col.lower()}"); panel(ax, L)
g = pd.Series({k: v["n"] for k, v in pd.DataFrame(rg_rows).set_index("Reason group").T.to_dict().items()}).sort_values()
cols = [C["grey"] if k in ["Unknown", "Uncontrollable"] else C["red"] for k in g.index]
axes[2].barh(g.index, g.values, color=cols, height=0.6)
for i, v in enumerate(g.values): axes[2].text(v + 8, i, f"{v} ({100*v/(y==1).sum():.0f}%)", va="center", fontsize=8)
axes[2].set_xlim(0, g.max() * 1.3); axes[2].set_xlabel("Churned customers")
axes[2].set_title("Stated churn reason groups"); panel(axes[2], "c")
savefig("Fig_2_data_overview")

# ---- Fig 3: predictive performance ----
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
bp = axes[0].boxplot([auc_mat[m] for m in order], vert=False, patch_artist=True, widths=0.55,
                     medianprops=dict(color=C["dark"]))
for patch, m in zip(bp["boxes"], order): patch.set_facecolor(MODEL_COL.get(m, C["grey"])); patch.set_alpha(0.55)
axes[0].set_yticks(range(1, len(order) + 1)); axes[0].set_yticklabels(order); axes[0].invert_yaxis()
axes[0].set_xlabel("AUC (outer folds)"); axes[0].set_title(f"Discrimination across {J} outer folds"); panel(axes[0], "a")
k_mod = len(order); Q = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}
CD = Q.get(k_mod, 2.85) * np.sqrt(k_mod * (k_mod + 1) / (6 * J))
ax = axes[1]; rk = ranks.sort_values()
ax.hlines(0, 1, k_mod, color=C["dark"], lw=1)
for i, (m, r) in enumerate(rk.items()):
    ax.plot(r, 0, "o", color=MODEL_COL.get(m, C["grey"]), ms=7)
    yy = 0.35 + 0.28 * (i % 3)
    ax.annotate(f"{m}\n{r:.2f}", (r, 0), xytext=(r, yy), ha="center", fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.5))
ax.plot([1, 1 + CD], [-0.45, -0.45], color=C["red"], lw=3); ax.text(1 + CD / 2, -0.75, f"CD = {CD:.2f}", ha="center", fontsize=8)
ax.set_xlim(0.7, k_mod + 0.3); ax.set_ylim(-1.0, 1.3); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("Mean rank (lower is better)"); ax.set_title("Critical-difference diagram (Nemenyi, α = .05)"); panel(ax, "b")
savefig("Fig_3_performance")

# ---- Fig 4: explanations & Rashomon ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
gis = gi.div(gi.sum(0), axis=1).sort_values(PRIMARY, ascending=False).head(12).iloc[::-1]
h = 0.8 / len(mods)
for i, m in enumerate(mods):
    axes[0].barh(np.arange(len(gis)) + (i - (len(mods) - 1) / 2) * h, gis[m] * 100, height=h, color=MODEL_COL[m], label=m)
axes[0].set_yticks(range(len(gis))); axes[0].set_yticklabels(gis.index)
axes[0].set_xlabel("Share of total |SHAP| (%)"); axes[0].legend(frameon=False, loc="lower right")
axes[0].set_title("Global importance (top 12)"); panel(axes[0], "a")
M = pd.DataFrame(np.eye(len(mods)), index=mods, columns=mods)
for r in st_rows[:-1]:
    a, b = r["Comparison"].split(" vs ")
    M.loc[a, b] = r["Global Kendall τ"]; M.loc[b, a] = r["Local top-3 Jaccard (mean)"]
sns.heatmap(M, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axes[1], annot_kws={"size": 9})
axes[1].set_title("Agreement: upper = global τ, lower = local Jaccard"); panel(axes[1], "b")
savefig("Fig_4_explanations_rashomon")

# ---- Fig 5: KEY — explanations vs stated reasons ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
sh_all = shares(SV[PRIMARY][y == 1], MAPS["Primary"]); sh_all["g"] = rg[y == 1]
br = sh_all.groupby("g")[list(MAPS["Primary"])].mean().mul(100)
br = br.loc[[g for g in ["Price", "Support", "Competitor", "Quality", "Uncontrollable", "Unknown"] if g in br.index]]
sns.heatmap(br, annot=True, fmt=".1f", cmap="Greys", cbar_kws={"label": "% of positive SHAP"}, ax=axes[0], annot_kws={"size": 8})
axes[0].set_ylabel("Stated reason group"); axes[0].set_xlabel("Explanation domain")
axes[0].set_title(f"{PRIMARY}: same explanation profile for every reason"); panel(axes[0], "a")
fp = FOREST[FOREST.Map == "Primary"].reset_index(drop=True)
ylab, ypos = [], []
for i, (_, r) in enumerate(fp.iterrows()):
    yv = len(fp) - i
    axes[1].errorbar(r.AUC, yv, xerr=[[r.AUC - r.lo], [r.hi - r.AUC]], fmt="o", color=DOM_COL[r.Domain], ms=4, capsize=2, lw=1)
    ylab.append(f"{r.Explanation} · {r.Domain}"); ypos.append(yv)
axes[1].axvline(0.5, color=C["dark"], ls="--", lw=0.8)
axes[1].set_yticks(ypos); axes[1].set_yticklabels(ylab, fontsize=7)
axes[1].set_xlabel("Domain AUC vs stated reason (95% CI)"); axes[1].set_xlim(0.40, 0.65)
axes[1].set_title("No explanation identifies the stated reason"); panel(axes[1], "b")
savefig("Fig_5_reason_validation")

# ---- Fig 6: positive control ----
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
real = FOREST[(FOREST.Explanation == f"SHAP-{PRIMARY}")]
for ax, mp_name, L in zip(axes, ["Primary", "Robustness"], "ab"):
    sub = PC[PC.Map == mp_name]
    for d in DOMS:
        g = sub[sub.Domain == d].groupby("signal")["AUC"]
        ax.plot(SIGNALS, g.mean(), "o-", color=DOM_COL[d], ms=3.5, lw=1.4, label=d)
        ax.fill_between(SIGNALS, g.quantile(.025), g.quantile(.975), color=DOM_COL[d], alpha=0.12)
        rv = real[(real.Map == mp_name) & (real.Domain == d)]["AUC"]
        if len(rv): ax.axhline(rv.iloc[0], color=DOM_COL[d], ls=":", lw=1.1)
    ax.axhline(0.5, color=C["grey"], lw=0.8)
    ax.set_xlabel("Injected reason–feature signal strength")
    ax.set_title(f"{mp_name} domain map" + (" (Total Charges → Relationship)" if mp_name == "Robustness" else ""))
    panel(ax, L)
axes[0].set_ylabel("Domain AUC"); axes[0].legend(frameon=False, title="Domain (dotted = real reasons)", loc="upper left")
savefig("Fig_6_positive_control")

# ---- Fig 7: whether vs why ----
t13 = pd.read_csv(ff("nb4/tables/T13_why_model_importance.csv"), index_col=0).iloc[:, 0].clip(lower=0)
why = t13 / t13.sum() if t13.sum() > 0 else t13
whe = gi[PRIMARY] / gi[PRIMARY].sum()
cmp_ = pd.DataFrame({"whether": whe, "why": why.reindex(FEATS).fillna(0)})
rho = stats.spearmanr(cmp_["whether"], cmp_["why"])[0]
top = cmp_.loc[cmp_.max(1).sort_values(ascending=False).index[:12]].iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
yy = np.arange(len(top))
axes[0].barh(yy + 0.2, top["whether"] * 100, 0.4, color=C["blue"], label=f"Churn model ({PRIMARY}, |SHAP|)")
axes[0].barh(yy - 0.2, top["why"] * 100, 0.4, color=C["red"], label="Why-model (permutation importance)")
axes[0].set_yticks(yy); axes[0].set_yticklabels(top.index); axes[0].set_xlabel("Share of importance (%)")
axes[0].legend(frameon=False, loc="lower right"); axes[0].set_title(f"Predicting WHETHER vs WHY (Spearman ρ = {rho:.2f})")
panel(axes[0], "a")
ab = t12.copy(); ab["mean"] = ab["Macro AUC"].str.split(" ± ").str[0].astype(float); ab["sd"] = ab["Macro AUC"].str.split(" ± ").str[1].astype(float)
sets = ["All features", "Without Latitude/Longitude", "Geography only"]
for j, (mdl, colr) in enumerate([("Multinomial LogReg", C["dark"]), ("Gradient Boosting", C["red"])]):
    s_ = ab[ab.Model == mdl].set_index("Feature set").loc[sets]
    axes[1].bar(np.arange(3) + (j - 0.5) * 0.35, s_["mean"], 0.35, yerr=s_["sd"], color=colr, alpha=0.8, label=mdl, capsize=3)
axes[1].axhline(0.5, ls="--", color=C["grey"], lw=0.8); axes[1].set_ylim(0.45, 0.65)
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["All", "No geography", "Geography only"])
axes[1].set_ylabel("Macro AUC (stated reason)"); axes[1].legend(frameon=False)
axes[1].set_title("Reason signal is almost entirely geographic"); panel(axes[1], "b")
savefig("Fig_7_whether_vs_why")

# ---- Fig 8: profit (recomputed from out-of-fold predictions, identical rules to NB5) ----
p_raw = SHAPM[PRIMARY]["prob"].values
pcal = np.zeros(N)
for k in np.unique(fold_of):
    pcal[fold_of == k] = IsotonicRegression(out_of_bounds="clip").fit(p_raw[fold_of != k], y[fold_of != k]).predict(p_raw[fold_of == k])
shap_dom = np.array(DOMS)[shares(SV[PRIMARY], MAPS["Primary"])[DOMS].values.argmax(1)]
maj_dom, why_dom = np.empty(N, object), np.empty(N, object)
for k in np.unique(fold_of):
    tr = (fold_of != k) & act; te = fold_of == k
    maj_dom[te] = pd.Series(true_dom[tr]).value_counts().index[0]
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("bin", "passthrough", BIN),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    w = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED))])
    why_dom[te] = w.fit(data.loc[tr, FEATS], true_dom[tr].astype(str)).predict(data.loc[te, FEATS])
unc, unk = rg == "Uncontrollable", rg == "Unknown"
def curve(score, od, gm=0.40, gx=0.10, cost=100):
    g = np.where(od == true_dom, gm, gx).astype(float); g[unk] = (gm + 2 * gx) / 3; g[unc] = 0; g[y == 0] = 0
    return np.concatenate([[0], np.cumsum((g * CLTV - cost)[np.argsort(-score, kind="stable")])])
sc = pcal * CLTV
curves = {"Majority offer (S2)": (curve(sc, maj_dom), C["blue"], "-"),
          "SHAP-guided offer (S3)": (curve(sc, shap_dom), C["red"], "-"),
          "Why-model offer (S4)": (curve(sc, why_dom), C["green"], "--"),
          "Oracle reason (S5, bound)": (curve(sc, true_dom), C["dark"], ":")}
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 1]})
x = np.arange(N + 1) / N * 100
for lab, (cv, col, ls) in curves.items():
    axes[0].plot(x, cv / 1e6, color=col, ls=ls, lw=1.6, label=f"{lab}: max {cv.max()/1e6:.2f}M")
axes[0].axhline(0, color=C["grey"], lw=0.8); axes[0].set_xlim(0, 100)
axes[0].set_xlabel("% of customers targeted (ranked by calibrated risk × CLTV)"); axes[0].set_ylabel("Cumulative profit (millions)")
axes[0].legend(frameon=True, framealpha=0.9, edgecolor="none", loc="lower right"); axes[0].set_title("Profit curves (out-of-fold)"); panel(axes[0], "a")
sens = pd.read_csv(ff("nb5/tables/T18_sensitivity_SHAP-guided.csv"), index_col=0)
vmax = np.nanmax(np.abs(sens.values))
sns.heatmap(sens, annot=True, fmt=".0f", cmap="RdBu", center=0, vmin=-vmax, vmax=vmax, cbar_kws={"label": "% vs majority offer"},
            ax=axes[1], annot_kws={"size": 8})
axes[1].set_title("SHAP-guided offers lose profit in every setting" if (sens.values < 0).all() else "SHAP-guided offer: % profit vs majority")
axes[1].set_xlabel("Contact cost"); axes[1].set_ylabel("Offer success if matched")
panel(axes[1], "b")
savefig("Fig_8_profit")

# =====================================================================
# EXPORT: Excel workbook + captions
# =====================================================================
xlsx = os.path.join(OUT, "manuscript_tables.xlsx")
with pd.ExcelWriter(xlsx) as w:
    for key, (df, cap, idx) in TABLES.items():
        pd.DataFrame({cap: []}).to_excel(w, sheet_name=key, index=False)
        df.to_excel(w, sheet_name=key, startrow=2, index=idx)

S = lambda k: curves[k][0].max()
pct = lambda a, b: 100 * (a - b) / abs(b)
fp_real = FOREST[(FOREST.Map == "Primary")]
FIGCAP = {
 "Fig_1": "Reason-validated XAI framework: (1) leakage-free risk model, (2) explanation, (3) validation against stated reasons with a positive control, (4) why-model, (5) profit evaluation. [Draw in draw.io]",
 "Fig_2": f"Data overview. (a, b) Churn rate by contract and internet service (dashed: overall {overall:.1f}%). (c) Stated reason groups among {int((y==1).sum())} churners; grey groups are excluded from validation.",
 "Fig_3": f"Predictive performance under nested cross-validation. (a) AUC over {J} outer folds. (b) Critical-difference diagram; models within CD = {CD:.2f} are not significantly different. {fr_txt}.",
 "Fig_4": f"Explanations differ across near-equally accurate models. (a) Global SHAP importance. (b) Agreement of explanations between models: global Kendall τ (upper triangle) and mean customer-level top-3 Jaccard (lower triangle).",
 "Fig_5": f"Explanations do not reflect stated reasons. (a) Mean share of positive {PRIMARY} attribution per domain is nearly identical across reason groups. (b) Domain AUCs with 95% bootstrap CIs (range {fp_real.AUC.min():.3f}–{fp_real.AUC.max():.3f}); 0.5 = no information.",
 "Fig_6": "Positive control. Synthetic reasons with known feature dependence are detected for Support and Offering as signal increases; dotted lines show real reasons. (b) Robustness map with Total Charges reassigned to the Relationship domain.",
 "Fig_7": f"Risk and reasons rely on different information. (a) Feature importance of the churn model vs the why-model (Spearman ρ = {rho:.2f}). (b) Recoverability of stated reasons collapses to chance without geography.",
 "Fig_8": f"Retention profit. (a) Profit curves: SHAP-guided offers {pct(S('SHAP-guided offer (S3)'), S('Majority offer (S2)')):+.1f}% and oracle reasons {pct(S('Oracle reason (S5, bound)'), S('Majority offer (S2)')):+.1f}% relative to the majority offer. (b) Sensitivity of the SHAP-guided loss to offer success and contact cost.",
}
with open(os.path.join(OUT, "captions.md"), "w") as f:
    f.write("# Draft captions (numbers filled from results)\n\n## Figures\n")
    for k, v in FIGCAP.items(): f.write(f"**{k.replace('_', ' ')}.** {v}\n\n")
    f.write("## Tables\n")
    for k, v in CAPTIONS.items(): f.write(f"**{k.replace('_', ' ')}.** {v}\n\n")

print("\n" + "=" * 70)
print("FINAL OUTPUT ->", OUT)
print("Figures:", sorted(f for f in os.listdir(FIG) if f.endswith(".png")))
print("Tables :", sorted(os.listdir(TAB)))
print("Excel  :", xlsx)
print("Captions: captions.md")
print("\nROBUSTNESS (Total Charges -> Relationship): see Table_6 rows 'Robustness' and Fig_6(b).")
print("Final tables and figures saved in", os.path.join(WORK_DIR, "final"))
