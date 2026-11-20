# =====================================================================
# NB4b (single cell): Where does the reason signal come from?
#   1. Recoverability with vs without geography (Latitude/Longitude)
#   2. Why-model feature importance (permutation importance, out-of-fold)
#   3. Price blind-spot diagnostic: does the churn model's Price-domain
#      SHAP share track Monthly Charges at all?
# Input : NB1, NB3 output in /kaggle/working     RUNTIME: ~5-8 min
# =====================================================================
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

SEED = 42
import os, json, glob, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

BASE = os.environ.get("NB_IN", WORK_DIR)
def find_file(name):
    for p in [os.path.join(BASE, name)] + glob.glob(f"{DATA_DIR}/**/{name}", recursive=True):
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb4")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})

cfg  = json.load(open(find_file("config.json")))
data = pd.read_csv(find_file("telco_clean.csv"))
NUM = [f for f in cfg["num_features"] if f != "Avg Monthly Spend"]
BIN, CAT = cfg["bin_features"], cfg["cat_features"]
GROUPS = ["Price", "Support", "Competitor", "Quality"]
ch = data[(data[cfg["target"]] == 1) & data["Reason Group"].isin(GROUPS)].reset_index(drop=True)
y4 = LabelEncoder().fit(GROUPS).transform(ch["Reason Group"])
GEO = ["Latitude", "Longitude"]

def make_pipe(num, model, bins=None, cats=None):
    bins = BIN if bins is None else bins; cats = CAT if cats is None else cats
    parts = [("num", StandardScaler(), num)]
    if bins: parts.append(("bin", "passthrough", bins))
    if cats: parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
    clf = (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          class_weight="balanced", random_state=SEED)
           if model == "GB" else LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
    return Pipeline([("pre", ColumnTransformer(parts)), ("clf", clf)])

CV = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(ch, y4))
def cv_auc(num, bins, cats, model):
    X = ch[num + bins + cats]
    return np.array([roc_auc_score(y4[te], make_pipe(num, model, bins, cats).fit(X.iloc[tr], y4[tr])
                                   .predict_proba(X.iloc[te]), multi_class="ovr", average="macro") for tr, te in CV])

# ---------------------------------------------------------------------
# 1. Recoverability with / without geography / geography only
# ---------------------------------------------------------------------
NUM_NOGEO = [f for f in NUM if f not in GEO]
sets = {"All features": (NUM, BIN, CAT),
        "Without Latitude/Longitude": (NUM_NOGEO, BIN, CAT),
        "Geography only": (GEO, [], [])}
rows, res = [], {}
for name, (num, bins, cats) in sets.items():
    for model in ["LogReg", "GB"]:
        a_ = cv_auc(num, bins, cats, model)
        res[(name, model)] = a_
        rows.append({"Feature set": name, "Model": "Gradient Boosting" if model == "GB" else "Multinomial LogReg",
                     "Macro AUC": f"{a_.mean():.3f} ± {a_.std():.3f}"})
geo_tab = pd.DataFrame(rows)
# paired test: all vs without geography (GB), same folds
d = res[("All features", "GB")] - res[("Without Latitude/Longitude", "GB")]
t, p = stats.ttest_rel(res[("All features", "GB")], res[("Without Latitude/Longitude", "GB")])
geo_tab.to_csv(os.path.join(TAB, "T12_recoverability_geo_ablation.csv"), index=False)
print("=== Table 12: Reason recoverability — role of geography (4 groups) ==="); display(geo_tab)
print(f"GB: all vs without geography  ΔAUC = {d.mean():.3f}, paired t = {t:.2f}, p = {p:.4f} (fold overlap -> indicative only)")

# ---------------------------------------------------------------------
# 2. Why-model importance (GB, all features): out-of-fold permutation importance
# ---------------------------------------------------------------------
X = ch[NUM + BIN + CAT]
imp = []
for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y4)):
    pipe = make_pipe(NUM, "GB").fit(X.iloc[tr], y4[tr])
    pi = permutation_importance(pipe, X.iloc[te], y4[te], n_repeats=10, random_state=SEED,
                                scoring=lambda est, Xv, yv: roc_auc_score(yv, est.predict_proba(Xv), multi_class="ovr", average="macro"))
    imp.append(pd.Series(pi.importances_mean, index=X.columns))
imp = pd.concat(imp, axis=1)
why_imp = pd.DataFrame({"Mean ΔAUC when permuted": imp.mean(1), "SD across folds": imp.std(1)}).sort_values(
    "Mean ΔAUC when permuted", ascending=False)
why_imp.round(4).to_csv(os.path.join(TAB, "T13_why_model_importance.csv"))
print("\n=== Table 13: What does the WHY-model use? (out-of-fold permutation importance) ===")
display(why_imp.head(10).round(4))

# churn-model importance for comparison (NB3)
for m_ in ["CatBoost", "XGBoost", "LogReg"]:
    try:
        shap_file = find_file(f"nb3/shap_{m_}.csv"); break
    except FileNotFoundError: pass
churn_model = os.path.basename(shap_file)[5:-4]
sv_all = pd.read_csv(shap_file)
feats = NUM + BIN + CAT
churn_imp = sv_all[feats].abs().mean(); churn_imp /= churn_imp.sum()
why_pos = why_imp["Mean ΔAUC when permuted"].clip(lower=0); why_pos /= why_pos.sum() if why_pos.sum() > 0 else 1
comp = pd.DataFrame({"Churn model (WHETHER)": churn_imp, "Why-model (WHY)": why_pos}).fillna(0)
comp = comp.loc[comp.max(1).sort_values(ascending=False).index[:12]]
rho = stats.spearmanr(churn_imp[feats], why_pos.reindex(feats).fillna(0))[0]

fig, ax = plt.subplots(figsize=(8, 5.5))
yy = np.arange(len(comp))[::-1]
ax.barh(yy + 0.2, comp.iloc[:, 0] * 100, height=0.4, color="#4A6FA5", label=f"Churn model ({churn_model} |SHAP| share)")
ax.barh(yy - 0.2, comp.iloc[:, 1] * 100, height=0.4, color="#C8553D", label="Why-model (permutation importance share)")
ax.set_yticks(yy); ax.set_yticklabels(comp.index); ax.set_xlabel("Share of importance (%)")
ax.set_title(f"Features that predict WHETHER ≠ features that inform WHY (Spearman ρ = {rho:.2f})")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F17_whether_vs_why.png"), bbox_inches="tight"); plt.show()
print(f"Rank correlation churn-importance vs why-importance: Spearman rho = {rho:.3f}")

# ---------------------------------------------------------------------
# 3. Price blind spot: does Price-domain SHAP share follow Monthly Charges?
# ---------------------------------------------------------------------
idx = data.index[(data[cfg["target"]] == 1) & data["Reason Group"].isin(GROUPS)].values
SV = sv_all.loc[idx, feats].values
pos = np.clip(SV, 0, None); tot = pos.sum(1); tot[tot == 0] = 1
share = lambda fs: pos[:, [feats.index(f) for f in fs]].sum(1) / tot
mc = data.loc[idx, "Monthly Charges"].values
diag = pd.DataFrame({
    "Spearman ρ with Monthly Charges": {
        "Price share (MC + TC + Gap)": stats.spearmanr(mc, share(["Monthly Charges", "Total Charges", "Charge Gap"]))[0],
        "SHAP of Monthly Charges":     stats.spearmanr(mc, SV[:, feats.index("Monthly Charges")])[0],
        "SHAP of Total Charges":       stats.spearmanr(mc, SV[:, feats.index("Total Charges")])[0],
        "SHAP of Tenure Months":       stats.spearmanr(mc, SV[:, feats.index("Tenure Months")])[0],
    }})
diag.round(3).to_csv(os.path.join(TAB, "T14_price_blindspot.csv"))
print("\n=== Table 14: Price blind-spot diagnostic (actionable churners) ==="); display(diag.round(3))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].scatter(mc, SV[:, feats.index("Monthly Charges")], s=4, alpha=0.4, color="#4A6FA5")
axes[0].set(xlabel="Monthly Charges", ylabel="SHAP (Monthly Charges)", title="Own-feature attribution")
axes[1].scatter(mc, share(["Monthly Charges", "Total Charges", "Charge Gap"]), s=4, alpha=0.4, color="#C8553D")
axes[1].set(xlabel="Monthly Charges", ylabel="Price-domain share", title="Price-domain share of churn credit")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F18_price_blindspot.png"), bbox_inches="tight"); plt.show()
print("\nSaved:", sorted(os.listdir(TAB)))
