# =====================================================================
# NB7b (single cell): ROBUSTNESS - counterfactuals WITHOUT contract/payment/paperless changes
# (answers: 'counterfactuals point to Contract only because the search favours it')
# NB7 (base): Counterfactual explanations vs stated reasons
#                    + statistical power of the validation
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
#
# Why: prior work prescribes retention actions with counterfactuals
#      (DiCE-style). Reviewers will ask whether the null result holds
#      for counterfactuals, not only SHAP/LIME.
#   A. Minimal-change counterfactuals: for every validation churner, the
#      smallest set (1 or 2) of ACTIONABLE feature changes that moves the
#      out-of-fold churn model below the decision threshold (exhaustive
#      search -> deterministic, reproducible)
#   B. Do counterfactual domains match stated reasons? (same metrics as NB3)
#   C. Profit of counterfactual-guided offers (same engine as NB5)
#   D. Power: minimum detectable domain AUC for the validation sample
#
# Input : NB1-NB5 output in /kaggle/working     RUNTIME: ~5-10 min
# Output: /kaggle/working/nb7/
# =====================================================================
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

EXCLUDE = {"Contract", "Payment Method", "Paperless Billing"}   # Relationship levers removed
THRESH  = 0.5      # counterfactual must bring raw churn probability below this
N_PERM  = 2000
N_BOOT  = 1000
SEED    = 42
G_MATCH, G_MISS, COST = 0.40, 0.10, 100

import os, json, glob, itertools, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, cohen_kappa_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

HAS = {}
try:
    from catboost import CatBoostClassifier; HAS["CatBoost"] = True
except ImportError: pass
try:
    from xgboost import XGBClassifier; HAS["XGBoost"] = True
except ImportError: pass

BASE = os.environ.get("NB_IN", WORK_DIR)
def ff(name):
    for p in [os.path.join(BASE, name)] + glob.glob(f"{DATA_DIR}/**/{name}", recursive=True):
        if os.path.exists(p): return p
    raise FileNotFoundError(f"{name} not found - run NB1-NB5 first")
OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb7b")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "pdf.fonttype": 42})

cfg    = json.load(open(ff("config.json")))
data   = pd.read_csv(ff("telco_clean.csv"))
splits = [s for s in json.load(open(ff("nb2/outer_splits.json"))) if s["repeat"] == 0]
bparam = pd.read_csv(ff("nb2/best_params.csv"))
NUM = [f for f in cfg["num_features"] if f != "Avg Monthly Spend"]
BIN, CAT = cfg["bin_features"], cfg["cat_features"]
FEATS = NUM + BIN + CAT
X, y = data[FEATS].copy(), data[cfg["target"]].astype(int).values
CLTV, N = data[cfg["value_col"]].astype(float).values, len(data)
fold_of = np.empty(N, int)
for s in splits: fold_of[s["test_idx"]] = s["fold"]

PRIMARY = next(m for m in ["CatBoost", "XGBoost", "LogReg"] if (m == "LogReg" or m in HAS) and (bparam.Model == m).any())
print("Counterfactuals generated for:", PRIMARY)

# ---------------------------------------------------------------------
# Reason ground truth + domains (identical to NB3/NB6 primary map)
# ---------------------------------------------------------------------
R2D = {"Price": "Price", "Support": "Support", "Competitor": "Offering", "Quality": "Offering"}
DOMS = ["Price", "Support", "Offering"]
FEAT_DOMAIN = {**{f: "Price" for f in ["Monthly Charges", "Total Charges", "Charge Gap"]},
               **{f: "Support" for f in ["Tech Support", "Online Security", "Device Protection"]},
               **{f: "Offering" for f in ["Internet Service", "Streaming TV", "Streaming Movies", "Multiple Lines",
                                          "Online Backup", "Phone Service", "Num Services"]},
               **{f: "Relationship" for f in ["Contract", "Tenure Months", "Paperless Billing", "Payment Method"]}}
rg = data["Reason Group"].astype(object).values
true_dom = np.array([R2D.get(r) if isinstance(r, str) else None for r in rg], dtype=object)
act = (y == 1) & np.isin(rg, list(R2D))
val_idx = np.where(act)[0]
print(f"Validation churners: {len(val_idx)}")

# ---------------------------------------------------------------------
# Refit out-of-fold primary model (NB2 tuned params, as in NB3)
# ---------------------------------------------------------------------
INT_P = {"n_estimators", "max_depth", "min_samples_leaf", "num_leaves", "min_child_samples", "min_child_weight", "iterations", "depth"}
def get_params(m, k):
    r = bparam[(bparam.Model == m) & (bparam.Fold == k)].iloc[0].to_dict()
    return {kk: (int(v) if kk in INT_P else v) for kk, v in r.items() if kk not in ("Model", "Fold") and not (isinstance(v, float) and np.isnan(v))}
def build(m, p, seed, pos_w):
    if m == "CatBoost":
        return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=float(p["learning_rate"]),
                                  l2_leaf_reg=float(p["l2_leaf_reg"]), auto_class_weights="Balanced", random_seed=seed, verbose=0, thread_count=-1)
    if m == "XGBoost":
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=float(p["learning_rate"]),
                             subsample=float(p["subsample"]), colsample_bytree=float(p["colsample_bytree"]), min_child_weight=p["min_child_weight"],
                             reg_lambda=float(p["reg_lambda"]), scale_pos_weight=pos_w, eval_metric="logloss", tree_method="hist",
                             n_jobs=-1, random_state=seed, verbosity=0)
    return LogisticRegression(C=float(p["C"]), class_weight="balanced", max_iter=3000)
def pipe_for(k):
    tr = fold_of != k
    pos_w = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("bin", "passthrough", BIN),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    return Pipeline([("pre", pre), ("clf", build(PRIMARY, get_params(PRIMARY, k), SEED + k, pos_w))]).fit(X[tr], y[tr])

# ---------------------------------------------------------------------
# A. Actionable changes (immutable: tenure, demographics, location, history)
# ---------------------------------------------------------------------
ADDONS = ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
CONTRACTS = sorted(X["Contract"].unique()); PAYMENTS = sorted(X["Payment Method"].unique())
def candidate_changes(row):
    """List of (feature, new_value) single changes that are feasible for this customer."""
    ch = [("Contract", v) for v in CONTRACTS if v != row["Contract"]]
    ch += [("Payment Method", v) for v in PAYMENTS if v != row["Payment Method"]]
    ch += [("Paperless Billing", 1 - row["Paperless Billing"])]
    if row["Internet Service"] != "No":
        ch += [(a, "Yes" if row[a] == "No" else "No") for a in ADDONS]
        ch += [("Internet Service", v) for v in ["DSL", "Fiber optic"] if v != row["Internet Service"]]
    if row["Phone Service"] == 1:
        ch += [("Multiple Lines", "Yes" if row["Multiple Lines"] == "No" else "No")]
    ch += [("Monthly Charges", f) for f in (0.9, 0.8, 0.7)]          # price reduction of 10/20/30%
    return [c for c in ch if c[0] not in EXCLUDE]
def apply(row, changes):
    r = row.copy()
    for f, v in changes:
        if f == "Monthly Charges":
            delta = r["Monthly Charges"] * (1 - v)
            r["Monthly Charges"] -= delta; r["Charge Gap"] -= delta
        else:
            r[f] = v
    r["Num Services"] = sum(r[a] == "Yes" for a in ADDONS) + r["Phone Service"] + int(r["Multiple Lines"] == "Yes")
    return r

t0 = time.time(); CF = {}
for k in np.unique(fold_of):
    pipe = pipe_for(k)
    ids = val_idx[fold_of[val_idx] == k]
    rows, meta = [], []
    for i in ids:
        row = X.iloc[i]; ch = candidate_changes(row)
        combos = [(c,) for c in ch] + [c for c in itertools.combinations(ch, 2) if c[0][0] != c[1][0]]
        for c in combos:
            rows.append(apply(row, c)); meta.append((i, c))
    P = pipe.predict_proba(pd.DataFrame(rows)[FEATS])[:, 1]
    p0 = pipe.predict_proba(X.iloc[ids])[:, 1]
    base = dict(zip(ids, p0))
    df = pd.DataFrame({"i": [m[0] for m in meta], "changes": [m[1] for m in meta], "p": P})
    df["n"] = df["changes"].map(len); df["flip"] = df["p"] < THRESH
    for i, g in df.groupby("i"):
        flips = g[g.flip]
        pick = (flips.sort_values(["n", "p"]).iloc[0] if len(flips) else g.sort_values(["p", "n"]).iloc[0])
        CF[i] = {"changes": pick.changes, "p_before": base[i], "p_after": pick.p, "flipped": bool(len(flips)), "n_changes": pick.n}
    print(f"fold {k+1}: {len(ids)} churners, {len(df):,} candidates | {(time.time()-t0)/60:.1f} min")

def cf_domains(changes):
    return [FEAT_DOMAIN[f] for f, _ in changes]
cf = pd.DataFrame.from_dict(CF, orient="index").loc[val_idx]
cf["domains"] = cf["changes"].map(cf_domains)
cf["features"] = cf["changes"].map(lambda c: " + ".join(f for f, _ in c))
cf["indicated"] = cf["domains"].map(lambda ds: next((d for d in ds if d in DOMS), "Relationship"))
cf["true"] = true_dom[val_idx]
cf.drop(columns=["changes"]).assign(domains=cf["domains"].map("|".join)).to_csv(os.path.join(OUT, "counterfactuals.csv"))

summ = pd.Series({
    "Churners": len(cf), "Flipped below threshold (%)": 100 * cf.flipped.mean(),
    "Single-change CFs (%)": 100 * (cf.n_changes == 1).mean(),
    "Mean churn prob. before": cf.p_before.mean(), "Mean churn prob. after": cf.p_after.mean(),
    "CFs touching only Relationship features (%)": 100 * (cf.indicated == "Relationship").mean()}).round(3)
top_cf = cf["features"].value_counts().head(10).rename("Customers").to_frame()
top_cf["%"] = (100 * top_cf["Customers"] / len(cf)).round(1)
summ.to_csv(os.path.join(TAB, "T20_cf_summary.csv")); top_cf.to_csv(os.path.join(TAB, "T21_cf_top_changes.csv"))
print("\n=== Counterfactual summary ==="); display(summ.to_frame("Value"))
print("\n=== Most frequent minimal counterfactuals ==="); display(top_cf)

# ---------------------------------------------------------------------
# B. Do counterfactual domains match stated reasons?
# ---------------------------------------------------------------------
td, ind = cf["true"].values, cf["indicated"].values
acc = (ind == td).mean()
perm = np.array([(ind == rng.permutation(td)).mean() for _ in range(N_PERM)])
row = {"Explanation": f"Counterfactual-{PRIMARY}", "Alignment acc.": round(acc, 3),
       "Permutation p": round((np.sum(perm >= acc) + 1) / (N_PERM + 1), 3),
       "Majority baseline": round(pd.Series(td).value_counts(normalize=True).iloc[0], 3),
       "Cohen κ": round(cohen_kappa_score(td, ind), 3)}
for d in DOMS:
    lab = (td == d).astype(int)
    sc = cf["domains"].map(lambda ds: ds.count(d) / len(ds)).values
    auc = roc_auc_score(lab, sc)
    bs = [roc_auc_score(lab[b], sc[b]) for b in (rng.integers(0, len(lab), len(lab)) for _ in range(N_BOOT)) if lab[b].min() != lab[b].max()]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    row[f"AUC {d} [95% CI]"] = f"{auc:.3f} [{lo:.3f}, {hi:.3f}]"
row["Relationship-only share"] = round((ind == "Relationship").mean(), 3)
cf_val = pd.DataFrame([row]).set_index("Explanation")
cf_val.to_csv(os.path.join(TAB, "T22_cf_reason_validation.csv"))
print("\n=== Counterfactual explanations vs stated reasons ==="); display(cf_val)
ct = pd.crosstab(pd.Series(td, name="Stated domain"), pd.Series(ind, name="Counterfactual domain"), normalize="index").mul(100).round(1)
ct.to_csv(os.path.join(TAB, "T23_cf_confusion.csv")); display(ct)

# ---------------------------------------------------------------------
# C. Profit of counterfactual-guided offers (NB5 engine, out-of-fold)
# ---------------------------------------------------------------------
p_raw = pd.read_csv(ff(f"nb3/shap_{PRIMARY}.csv"))["prob"].values
pcal = np.zeros(N)
for k in np.unique(fold_of):
    pcal[fold_of == k] = IsotonicRegression(out_of_bounds="clip").fit(p_raw[fold_of != k], y[fold_of != k]).predict(p_raw[fold_of == k])
maj = np.empty(N, object)
for k in np.unique(fold_of):
    maj[fold_of == k] = pd.Series(true_dom[(fold_of != k) & act]).value_counts().index[0]
unc, unk = rg == "Uncontrollable", rg == "Unknown"
def curve(od, idx=None, gm=G_MATCH, gx=G_MISS, cost=COST):
    idx = np.arange(N) if idx is None else idx
    g = np.where(od == true_dom, gm, gx).astype(float); g[unk] = (gm + 2 * gx) / 3; g[unc] = 0; g[y == 0] = 0
    sc = (pcal * CLTV)[idx]
    return np.concatenate([[0], np.cumsum((g * CLTV - cost)[idx][np.argsort(-sc, kind="stable")])]).max()
cf_contract = maj.copy(); cf_contract[val_idx] = cf["indicated"].values          # Relationship -> retention-contract offer (matches no reason)
cf_fallback = cf_contract.copy(); rel = val_idx[cf["indicated"].values == "Relationship"]; cf_fallback[rel] = maj[rel]
prow = []
for name, od in [("S2 Majority offer", maj), ("S6r Restricted counterfactual offer", cf_contract), ("S6rf Restricted counterfactual, majority fallback", cf_fallback)]:
    mp = curve(od); diffs = []
    for _ in range(N_BOOT):
        b = rng.integers(0, N, N); diffs.append(curve(od, b) - curve(maj, b))
    prow.append({"Strategy": name, "Max profit": round(mp), "% vs S2": round(100 * (mp - curve(maj)) / curve(maj), 1),
                 "Δ vs S2 [95% CI]": "—" if name.startswith("S2") else f"{np.mean(diffs):,.0f} [{np.percentile(diffs, 2.5):,.0f}, {np.percentile(diffs, 97.5):,.0f}]",
                 "Offer–reason match": round((od[val_idx] == true_dom[val_idx]).mean(), 3)})
cf_profit = pd.DataFrame(prow).set_index("Strategy")
cf_profit.to_csv(os.path.join(TAB, "T24_cf_profit.csv"))
print("\n=== Profit of counterfactual-guided offers ==="); display(cf_profit)

# ---------------------------------------------------------------------
# D. Power: minimum detectable domain AUC (Hanley & McNeil 1982), alpha=.05 two-sided, power .80
# ---------------------------------------------------------------------
def se_auc(A, n1, n0):
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    return np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A)) / (n1 * n0))
pw = []
for d in DOMS:
    n1 = int((td == d).sum()); n0 = len(td) - n1
    grid = np.arange(0.5005, 0.7, 0.0005)
    mde = next(A for A in grid if (A - 0.5) >= (1.96 * se_auc(0.5, n1, n0) + 0.8416 * se_auc(A, n1, n0)))
    pw.append({"Domain": d, "Churners stating it": n1, "Others": n0, "Minimum detectable AUC (80% power)": round(mde, 3)})
power = pd.DataFrame(pw).set_index("Domain")
power.to_csv(os.path.join(TAB, "T25_power.csv"))
print("\n=== Power of the domain-AUC test ==="); display(power)

# ---------------------------------------------------------------------
# Figure: where counterfactuals point vs what customers said
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.2))
cols = ["Price", "Support", "Offering", "Relationship"]
ct2 = ct.reindex(columns=cols, fill_value=0)
left = np.zeros(len(ct2))
pal = {"Price": "#C0392B", "Support": "#3B5B92", "Offering": "#4E8A55", "Relationship": "#8C939D"}
for c in cols:
    ax.barh(ct2.index, ct2[c], left=left, color=pal[c], label=c, height=0.6); left += ct2[c].values
ax.set_xlabel("% of churners"); ax.set_ylabel("Stated reason domain"); ax.set_xlim(0, 100)
ax.legend(title="Counterfactual changes", frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=8)
plt.tight_layout()
for ext in ["png", "pdf"]: plt.savefig(os.path.join(FIG, f"Fig_S1_counterfactual_domains_restricted.{ext}"), bbox_inches="tight")
plt.show()
print("\nSaved to", OUT, "->", sorted(os.listdir(TAB)))
