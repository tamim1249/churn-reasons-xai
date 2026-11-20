# =====================================================================
# NB3 (single cell): Explainability + Reason Validation  (MAIN NOVELTY)
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
#
# RQ2: Do SHAP / LIME explanations agree with customers' own stated
#      churn reasons?
# Solves: L2 (explanations never validated against ground truth),
#         L5 (no stability analysis), L8 (subjective evaluation)
#
# Input : NB1 + NB2 output in /kaggle/working (same notebook)
# Output: /kaggle/working/nb3/ -> SHAP values, tables, figures
# RUNTIME: ~5-15 min (LIME part ~5-10 min; set RUN_LIME=False to skip)
# =====================================================================

import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

RUN_LIME     = True
LIME_SAMPLES = 1000      # perturbations per explained customer
N_PERM       = 5000      # permutation-test iterations
N_BOOT       = 1000      # bootstrap iterations for CIs
SEED         = 42

import os, json, glob, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

HAS = {}
try:
    import xgboost as xgb; from xgboost import XGBClassifier; HAS["XGBoost"] = True
except ImportError: print("xgboost not found")
try:
    from catboost import CatBoostClassifier, Pool; HAS["CatBoost"] = True
except ImportError: print("catboost not found")
try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    if RUN_LIME: print("lime not found -> run `!pip install lime` in a cell above, or LIME will be skipped")
    RUN_LIME = False

# ---------------------------------------------------------------------
# 1. Load NB1 + NB2 outputs
# ---------------------------------------------------------------------
BASE = os.environ.get("NB_IN", WORK_DIR)
def find_file(name):
    for p in [os.path.join(BASE, name)] + glob.glob(f"{DATA_DIR}/**/{name}", recursive=True):
        if os.path.exists(p): return p
    raise FileNotFoundError(f"{name} not found. Run NB1 and NB2 first.")

OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb3")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)

cfg    = json.load(open(find_file("config.json")))
data   = pd.read_csv(find_file("telco_clean.csv"))
splits = json.load(open(find_file("nb2/outer_splits.json")))
bparam = pd.read_csv(find_file("nb2/best_params.csv"))
TARGET = cfg["target"]

NUM = [f for f in cfg["num_features"] if f != "Avg Monthly Spend"]   # same as NB2
BIN, CAT = cfg["bin_features"], cfg["cat_features"]
FEATS = NUM + BIN + CAT
X, y = data[FEATS].copy(), data[TARGET].astype(int).values

# Repeat 0 folds: every customer is in exactly one test fold
FOLDS = [s for s in splits if s["repeat"] == 0]
assert sorted(np.concatenate([f["test_idx"] for f in FOLDS])) == list(range(len(data)))

MODELS = [m for m in ["CatBoost", "XGBoost", "LogReg"] if m == "LogReg" or m in HAS]
MODELS = [m for m in MODELS if (bparam.Model == m).any()]
PRIMARY = MODELS[0]
print("Models explained:", MODELS, "| primary:", PRIMARY)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight"); plt.show()

# ---------------------------------------------------------------------
# 2. Feature -> explanation domain, reason group -> domain
#    PRIMARY mapping + ALTERNATIVE mapping (sensitivity analysis)
# ---------------------------------------------------------------------
DOMAIN_MAP = {
    "Price":    ["Monthly Charges", "Total Charges", "Charge Gap"],
    "Support":  ["Tech Support", "Online Security", "Device Protection"],
    "Offering": ["Internet Service", "Streaming TV", "Streaming Movies", "Multiple Lines",
                 "Online Backup", "Phone Service", "Num Services"],
    "Relationship": ["Contract", "Tenure Months", "Paperless Billing", "Payment Method"],
    "Demographic":  ["Male", "Senior Citizen", "Partner", "Dependents", "Latitude", "Longitude"],
}
# Alternative: security/protection are product add-ons, not support; fibre speed is price-tier related
DOMAIN_MAP_ALT = {
    "Price":    ["Monthly Charges", "Total Charges", "Charge Gap", "Internet Service"],
    "Support":  ["Tech Support"],
    "Offering": ["Online Security", "Device Protection", "Streaming TV", "Streaming Movies",
                 "Multiple Lines", "Online Backup", "Phone Service", "Num Services"],
    "Relationship": DOMAIN_MAP["Relationship"],
    "Demographic":  DOMAIN_MAP["Demographic"],
}
for mp in [DOMAIN_MAP, DOMAIN_MAP_ALT]:
    assert sorted(f for v in mp.values() for f in v) == sorted(FEATS), "Every feature must be mapped once"

REASON_TO_DOMAIN = {"Price": "Price", "Support": "Support", "Competitor": "Offering", "Quality": "Offering"}
ACTIONABLE = ["Price", "Support", "Offering"]          # domains a stated reason can point to
ALL_DOMAINS = list(DOMAIN_MAP)

# ---------------------------------------------------------------------
# 3. Refit fold models (NB2 tuned params) and compute local explanations
#    SHAP: native TreeSHAP for CatBoost/XGBoost, exact linear SHAP for LogReg
#    (all in log-odds space). One-hot SHAP summed back to original feature.
# ---------------------------------------------------------------------
INT_P = {"n_estimators", "max_depth", "min_samples_leaf", "num_leaves", "min_child_samples",
         "min_child_weight", "iterations", "depth"}
def get_params(model, fold):
    r = bparam[(bparam.Model == model) & (bparam.Fold == fold)].iloc[0].to_dict()
    return {k: (int(v) if k in INT_P else v) for k, v in r.items()
            if k not in ("Model", "Fold") and not (isinstance(v, float) and np.isnan(v))}

def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM), ("bin", "passthrough", BIN),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])

def build(model, p, seed, pos_w):
    if model == "LogReg":
        return LogisticRegression(C=float(p["C"]), class_weight="balanced", max_iter=3000)
    if model == "XGBoost":
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                             learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
                             colsample_bytree=float(p["colsample_bytree"]), min_child_weight=p["min_child_weight"],
                             reg_lambda=float(p["reg_lambda"]), scale_pos_weight=pos_w, eval_metric="logloss",
                             tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0)
    if model == "CatBoost":
        return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=float(p["learning_rate"]),
                                  l2_leaf_reg=float(p["l2_leaf_reg"]), auto_class_weights="Balanced",
                                  random_seed=seed, verbose=0, thread_count=-1)

def output_to_feature(pre):
    """Original feature name for every transformed column."""
    owners = []
    for name, trans, cols in pre.transformers_:
        if name == "cat":
            for c, cats in zip(cols, trans.categories_): owners += [c] * len(cats)
        elif name in ("num", "bin"):
            owners += list(cols)
    return np.array(owners)

def local_shap(model, clf, Xt_test, Xt_train):
    if model == "CatBoost":
        sv = clf.get_feature_importance(Pool(Xt_test), type="ShapValues")[:, :-1]
    elif model == "XGBoost":
        sv = clf.get_booster().predict(xgb.DMatrix(Xt_test), pred_contribs=True)[:, :-1]
    else:  # linear SHAP (interventional, feature independence): coef * (x - E[x])
        sv = clf.coef_[0] * (Xt_test - Xt_train.mean(axis=0))
    return sv

SHAP = {m: np.zeros((len(data), len(FEATS))) for m in MODELS}
PROB = {m: np.zeros(len(data)) for m in MODELS}
FITTED = {}   # (model, fold) -> pipeline (reused by LIME and NB5)
t0 = time.time()
for f in FOLDS:
    k, te = f["fold"], np.array(f["test_idx"])
    tr = np.setdiff1d(np.arange(len(data)), te)
    pos_w = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    for m in MODELS:
        pipe = Pipeline([("pre", make_pre()), ("clf", build(m, get_params(m, k), SEED + k, pos_w))])
        pipe.fit(X.iloc[tr], y[tr])
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        Xt_te, Xt_tr = pre.transform(X.iloc[te]), pre.transform(X.iloc[tr])
        sv = local_shap(m, clf, Xt_te, Xt_tr)
        owners = output_to_feature(pre)
        agg = pd.DataFrame(sv, columns=owners).T.groupby(level=0).sum().T[FEATS].values   # sum one-hot back
        SHAP[m][te] = agg
        PROB[m][te] = pipe.predict_proba(X.iloc[te])[:, 1]
        FITTED[(m, k)] = pipe
    print(f"fold {k+1}/{len(FOLDS)} done | {(time.time()-t0)/60:.1f} min")

for m in MODELS:
    print(f"{m:9s} out-of-fold AUC (repeat 1) = {roc_auc_score(y, PROB[m]):.4f}")
    pd.DataFrame(SHAP[m], columns=FEATS).assign(CustomerID=data[cfg["id_col"]], prob=PROB[m]) \
      .to_csv(os.path.join(OUT, f"shap_{m}.csv"), index=False)

# ---------------------------------------------------------------------
# 4. Global importance + stability (L5)
# ---------------------------------------------------------------------
glob_imp = pd.DataFrame({m: np.abs(SHAP[m]).mean(axis=0) for m in MODELS}, index=FEATS)
glob_imp = glob_imp.div(glob_imp.sum(axis=0), axis=1)          # share of total |SHAP|
glob_imp = glob_imp.sort_values(PRIMARY, ascending=False)
glob_imp.round(4).to_csv(os.path.join(TAB, "T5_global_importance.csv"))

# (a) across folds within the primary model
fold_rank = {}
for f in FOLDS:
    te = np.array(f["test_idx"])
    fold_rank[f"Fold {f['fold']+1}"] = np.abs(SHAP[PRIMARY][te]).mean(axis=0)
fold_rank = pd.DataFrame(fold_rank, index=FEATS)
tau_folds = [stats.kendalltau(fold_rank.iloc[:, i], fold_rank.iloc[:, j])[0]
             for i in range(fold_rank.shape[1]) for j in range(i + 1, fold_rank.shape[1])]
# (b) across models
tau_models = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
for a in MODELS:
    for b in MODELS:
        tau_models.loc[a, b] = stats.kendalltau(glob_imp[a], glob_imp[b])[0]
# (c) local: top-3 feature overlap per customer between models (Jaccard)
def topk(sv, k=3): return np.argsort(-np.abs(sv), axis=1)[:, :k]
local_rows = []
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        ta, tb = topk(SHAP[a]), topk(SHAP[b])
        jac = [len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(ta, tb)]
        local_rows.append({"Pair": f"{a} vs {b}", "Global Kendall tau": tau_models.loc[a, b],
                           "Local top-3 Jaccard (mean)": np.mean(jac)})
local_rows.append({"Pair": f"{PRIMARY}: across {len(FOLDS)} folds", "Global Kendall tau": np.mean(tau_folds),
                   "Local top-3 Jaccard (mean)": np.nan})
stab = pd.DataFrame(local_rows)
stab.round(4).to_csv(os.path.join(TAB, "T6_explanation_stability.csv"), index=False)
print("\n=== Table 5: Global importance (share of total |SHAP|) ==="); display(glob_imp.round(4))
print("\n=== Table 6: Explanation stability ==="); display(stab.round(4))

# F11: global importance comparison
top = glob_imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
h = 0.8 / len(MODELS)
for i, m in enumerate(MODELS):
    ax.barh(np.arange(len(top)) + i * h, top[m] * 100, height=h, label=m)
ax.set_yticks(np.arange(len(top)) + h * (len(MODELS) - 1) / 2); ax.set_yticklabels(top.index)
ax.set_xlabel("Share of total |SHAP| (%)"); ax.legend(frameon=False)
ax.set_title("Global feature importance (top 15, out-of-fold SHAP)")
savefig("F11_global_importance")

# F12: SHAP beeswarm-style dot plot for primary model (numeric/binary features coloured by value)
top10 = glob_imp.index[:10][::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
for i, feat in enumerate(top10):
    j = FEATS.index(feat); sv = SHAP[PRIMARY][:, j]
    v = X[feat]
    col = (v.rank(pct=True).values if feat in NUM + BIN else
           pd.Series(pd.factorize(v)[0]).rank(pct=True).values)
    jitter = rng.normal(0, 0.12, len(sv))
    sc = ax.scatter(sv, i + jitter, c=col, cmap="coolwarm", s=3, alpha=0.5, rasterized=True)
ax.axvline(0, color="grey", lw=0.8); ax.set_yticks(range(len(top10))); ax.set_yticklabels(top10)
ax.set_xlabel("SHAP value (log-odds of churn)")
cb = plt.colorbar(sc, ax=ax, shrink=0.6); cb.set_label("Feature value (low → high / category code)")
ax.set_title(f"{PRIMARY}: local SHAP distribution (top 10)")
savefig("F12_shap_beeswarm")

# ---------------------------------------------------------------------
# 5. Optional LIME (primary model, churners with actionable stated reason)
#    LIME on the raw feature space with proper categorical handling
# ---------------------------------------------------------------------
churn_mask = (y == 1)
val_mask = churn_mask & data["Reason Group"].isin(REASON_TO_DOMAIN).values
val_idx = np.where(val_mask)[0]
print(f"\nValidation set: {len(val_idx)} churners with an actionable stated reason "
      f"(excluded: Uncontrollable + Unknown = {int(churn_mask.sum() - len(val_idx))})")

LIME = None
if RUN_LIME:
    cat_like = CAT
    enc = {c: {v: i for i, v in enumerate(sorted(X[c].unique()))} for c in cat_like}
    dec = {c: {i: v for v, i in enc[c].items()} for c in cat_like}
    Xcode = X.copy()
    for c in cat_like: Xcode[c] = X[c].map(enc[c])
    Xcode = Xcode.astype(float)
    cat_idx = [FEATS.index(c) for c in cat_like + BIN]
    cat_names = {FEATS.index(c): [dec[c][i] for i in range(len(dec[c]))] for c in cat_like}
    cat_names.update({FEATS.index(c): ["0", "1"] for c in BIN})

    def make_predict(pipe):
        def fn(arr):
            df = pd.DataFrame(arr, columns=FEATS)
            for c in cat_like: df[c] = df[c].round().clip(0, len(dec[c]) - 1).astype(int).map(dec[c])
            for c in BIN: df[c] = df[c].round().clip(0, 1).astype(int)
            return pipe.predict_proba(df)
        return fn

    LIME = np.zeros((len(data), len(FEATS)))
    t0 = time.time()
    fold_of = {i: f["fold"] for f in FOLDS for i in f["test_idx"]}
    for f in FOLDS:
        te = np.array(f["test_idx"]); tr = np.setdiff1d(np.arange(len(data)), te)
        explainer = LimeTabularExplainer(Xcode.iloc[tr].values, feature_names=FEATS, class_names=["stay", "churn"],
                                         categorical_features=cat_idx, categorical_names=cat_names,
                                         discretize_continuous=True, random_state=SEED, mode="classification")
        fn = make_predict(FITTED[(PRIMARY, f["fold"])])
        for i in np.intersect1d(te, val_idx):
            exp = explainer.explain_instance(Xcode.iloc[i].values, fn, num_features=len(FEATS),
                                             num_samples=LIME_SAMPLES, labels=(1,))
            for j, w in exp.as_map()[1]: LIME[i, j] = w
        print(f"LIME fold {f['fold']+1}/{len(FOLDS)} | {(time.time()-t0)/60:.1f} min")
    pd.DataFrame(LIME[val_idx], columns=FEATS).assign(CustomerID=data[cfg["id_col"]].values[val_idx]) \
      .to_csv(os.path.join(OUT, f"lime_{PRIMARY}.csv"), index=False)

# ---------------------------------------------------------------------
# 6. REASON VALIDATION (RQ2)
#    Domain score = share of a customer's POSITIVE (churn-pushing) attribution
#    that falls in each domain. Explanation "points to" the actionable domain
#    with the highest score. Compared with the customer's stated reason.
# ---------------------------------------------------------------------
true_dom = data.loc[val_idx, "Reason Group"].map(REASON_TO_DOMAIN).values

def domain_shares(att, dmap):
    pos = np.clip(att, 0, None)
    tot = pos.sum(axis=1, keepdims=True); tot[tot == 0] = 1
    return pd.DataFrame({d: pos[:, [FEATS.index(f) for f in fs]].sum(axis=1) / tot[:, 0]
                         for d, fs in dmap.items()})

def boot_ci(fn, n=N_BOOT):
    vals = []
    for _ in range(n):
        b = rng.integers(0, len(val_idx), len(val_idx))
        try: vals.append(fn(b))
        except ValueError: pass
    return np.percentile(vals, [2.5, 97.5])

def validate(att, label, dmap):
    sh = domain_shares(att[val_idx], dmap)
    pred = sh[ACTIONABLE].values.argmax(axis=1)
    pred_dom = np.array(ACTIONABLE)[pred]
    acc = (pred_dom == true_dom).mean()
    # permutation null: shuffle stated reasons across churners
    perm = np.array([(pred_dom == rng.permutation(true_dom)).mean() for _ in range(N_PERM)])
    p_perm = (np.sum(perm >= acc) + 1) / (N_PERM + 1)
    majority = pd.Series(true_dom).value_counts(normalize=True).iloc[0]
    row = {"Explanation": label, "Alignment acc.": acc, "Perm. null mean": perm.mean(),
           "Perm. p": p_perm, "Majority baseline": majority,
           "Cohen kappa": cohen_kappa_score(true_dom, pred_dom)}
    # threshold-free: does the domain score rank customers who state that reason higher?
    for d in ACTIONABLE:
        lab = (true_dom == d).astype(int)
        auc = roc_auc_score(lab, sh[d])
        lo, hi = boot_ci(lambda b: roc_auc_score(lab[b], sh[d].values[b]))
        row[f"AUC {d}"] = f"{auc:.3f} [{lo:.3f}, {hi:.3f}]"
    row["Non-actionable share"] = sh[["Relationship", "Demographic"]].sum(axis=1).mean()
    return row, sh, pred_dom, perm

results, store = [], {}
for m in MODELS:
    for mp_name, mp in [("primary", DOMAIN_MAP), ("alternative", DOMAIN_MAP_ALT)]:
        r, sh, pdm, perm = validate(SHAP[m], f"SHAP-{m} ({mp_name} map)", mp)
        results.append(r); store[(f"SHAP-{m}", mp_name)] = (sh, pdm, perm)
if LIME is not None:
    for mp_name, mp in [("primary", DOMAIN_MAP), ("alternative", DOMAIN_MAP_ALT)]:
        r, sh, pdm, perm = validate(LIME, f"LIME-{PRIMARY} ({mp_name} map)", mp)
        results.append(r); store[(f"LIME-{PRIMARY}", mp_name)] = (sh, pdm, perm)

val_tab = pd.DataFrame(results).set_index("Explanation")
num_cols = ["Alignment acc.", "Perm. null mean", "Perm. p", "Majority baseline", "Cohen kappa", "Non-actionable share"]
val_tab[num_cols] = val_tab[num_cols].astype(float).round(4)
val_tab.to_csv(os.path.join(TAB, "T7_reason_validation.csv"))
print("\n=== Table 7: Do explanations reflect stated churn reasons? ===")
display(val_tab)

# Domain shares by stated reason group (all churners, primary model, primary map)
sh_all = domain_shares(SHAP[PRIMARY][churn_mask], DOMAIN_MAP)
sh_all["Reason Group"] = data.loc[churn_mask, "Reason Group"].values
by_reason = sh_all.groupby("Reason Group")[ALL_DOMAINS].mean().mul(100)
by_reason = by_reason.loc[[g for g in ["Price", "Support", "Competitor", "Quality", "Uncontrollable", "Unknown"] if g in by_reason.index]]
by_reason.round(1).to_csv(os.path.join(TAB, "T8_domain_share_by_reason.csv"))
kw = {d: stats.kruskal(*[g[d].values for _, g in sh_all.groupby("Reason Group")]) for d in ALL_DOMAINS}
kw_tab = pd.DataFrame({d: {"H": r.statistic, "p": r.pvalue} for d, r in kw.items()}).T
kw_tab.round(5).to_csv(os.path.join(TAB, "T8b_kruskal_domain_by_reason.csv"))
print(f"\n=== Table 8: {PRIMARY} mean domain share (%) by stated reason ==="); display(by_reason.round(1))
print("Kruskal-Wallis (does domain share differ across reason groups?)"); display(kw_tab.round(5))

# ---------------------------------------------------------------------
# 7. Figures for the validation result
# ---------------------------------------------------------------------
# F13: heatmap — what explanations emphasise vs what customers said
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(by_reason, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "% of positive SHAP"}, ax=ax)
ax.set_title(f"Where {PRIMARY} explanations put churn credit, by customer's stated reason")
ax.set_xlabel("Explanation domain"); ax.set_ylabel("Stated reason group")
savefig("F13_domain_share_by_reason")

# F14: confusion matrix (primary model, primary map) + permutation null
sh, pdm, perm = store[(f"SHAP-{PRIMARY}", "primary")]
cm = pd.DataFrame(confusion_matrix(true_dom, pdm, labels=ACTIONABLE, normalize="true") * 100,
                  index=[f"Stated: {d}" for d in ACTIONABLE], columns=[f"Expl.: {d}" for d in ACTIONABLE])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Oranges", cbar=False, ax=axes[0])
axes[0].set_title("Stated reason vs explanation (row %)")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
acc = (pdm == true_dom).mean()
axes[1].hist(perm, bins=40, color="#4A6FA5", alpha=0.8)
axes[1].axvline(acc, color="#C8553D", lw=2, label=f"Observed = {acc:.3f}")
axes[1].set_xlabel("Alignment accuracy under shuffled reasons"); axes[1].set_ylabel("Count")
axes[1].set_title(f"Permutation null ({N_PERM} shuffles)"); axes[1].legend(frameon=False)
savefig("F14_alignment_confusion_permutation")

pd.Series({"n_validation": len(val_idx), "models": MODELS, "lime": LIME is not None}).to_json(os.path.join(OUT, "meta.json"))
print("\nSaved to", OUT, "->", sorted(os.listdir(OUT)))
print("\nHOW TO READ TABLE 7:")
print(" - Alignment acc. >> Perm. null mean and Perm. p < .05  -> explanations carry real reason signal")
print(" - AUC <domain> ~0.5 -> explanation cannot tell who left for that reason")
print(" - High 'Non-actionable share' -> explanations are dominated by risk markers (contract, tenure),")
print("   not by the causes customers report. BOTH outcomes are publishable findings.")
