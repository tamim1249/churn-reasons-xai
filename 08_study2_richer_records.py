# =====================================================================
# NB8 (single cell): STUDY 2 — Does richer CRM data close the gap?
# Data: IBM Telco Customer Churn 11.1.3+ FULL release (Kaggle:
#       alfathterry/telco-customer-churn-11-1-3). Same customers as Study 1,
#       but with billing and usage features that map DIRECTLY onto stated
#       reasons (extra data charges, long-distance charges, download volume,
#       unlimited data, offers, refunds).
# Tests:
#   A. Leakage audit incl. Satisfaction Score / Customer Status
#   B. Risk model (CatBoost + LogReg, stratified 5-fold, out-of-fold)
#   C. Domain-level reason validation (same protocol as Study 1)
#   D. Feature-level test: for reasons with a DIRECTLY matching feature,
#      (i) do these churners differ on that feature?  (ii) does SHAP
#      single out that feature for them?
#   E. Reason recoverability (why-model) with/without geography
# RUNTIME: ~5-10 min.  Output: /kaggle/working/nb8/
# =====================================================================
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

SEED, N_PERM, N_BOOT, K = 42, 2000, 1000, 5

import os, re, glob, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, cohen_kappa_score, average_precision_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
try:
    from catboost import CatBoostClassifier, Pool; HAS_CB = True
except ImportError:
    HAS_CB = False; print("catboost not found -> LogReg only")

OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb8"); TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
def save(df, name, index=True): df.to_csv(os.path.join(TAB, name + ".csv"), index=index); display(df)

# ---------------------------------------------------------------------
# 0. Load: largest csv/xlsx in an attached dataset whose path mentions 11-1-3 / 11.1.3
# ---------------------------------------------------------------------
DATA_PATH = ""   # optional: paste the exact file path here if auto-detection fails
# Auto-detect by COLUMNS (not by folder name): the 11.1.3+ release is the only one
# with billing columns such as "Total Extra Data Charges" / "Customer Status".
MARKERS = {"totalextradatacharges", "customerstatus", "avgmonthlygbdownload", "totallongdistancecharges", "churncategory"}
def header(p):
    try:
        h = pd.read_excel(p, nrows=0) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p, nrows=0)
        return {re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in h.columns}
    except Exception: return set()
files = [p for p in glob.glob(os.environ.get("NB8_GLOB", DATA_DIR + "/**/*"), recursive=True)
         if p.lower().endswith((".csv", ".xlsx", ".xls"))]
print(f"{len(files)} csv/xlsx files under {DATA_DIR}:"); [print("  ", p) for p in files]
if DATA_PATH:
    path = DATA_PATH
else:
    score = {p: len(header(p) & MARKERS) for p in files}
    best = [p for p, s in score.items() if s >= 2]
    assert best, ("No file with IBM 11.1.3+ columns found. In the right panel: + Add Input -> search "
                  "'telco customer churn 11.1.3' -> Add. Or set DATA_PATH to the file path printed above.")
    path = max(best, key=lambda p: (score[p], os.path.getsize(p)))
raw = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
print("Loaded:", path, raw.shape)

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
COL = {norm(c): c for c in raw.columns}
def col(*names, required=True):
    for n in names:
        if norm(n) in COL: return COL[norm(n)]
    if required: raise KeyError(f"None of {names} found. Columns: {list(raw.columns)}")
    return None

# Roles (first matching name wins). Missing optional columns are skipped.
C = dict(
    id=col("Customer ID", "CustomerID"), status=col("Customer Status", required=False),
    churn=col("Churn Label", "Churn", "Churn Value"), reason=col("Churn Reason"),
    category=col("Churn Category", required=False), score=col("Churn Score", required=False),
    satisfaction=col("Satisfaction Score", required=False), cltv=col("CLTV", required=False))
y_raw = raw[C["churn"]]
y = (y_raw.astype(str).str.lower().isin(["yes", "1", "true", "churned"])).astype(int).values
print(f"Churn rate: {y.mean():.4f} | churners: {y.sum()}")

DOMAIN_CANDIDATES = {   # feature -> domain (only those present are used)
    "Price": ["Monthly Charge", "Monthly Charges", "Total Charges", "Total Extra Data Charges", "Total Long Distance Charges",
              "Avg Monthly Long Distance Charges", "Total Refunds", "Offer"],
    "Support": ["Premium Tech Support", "Tech Support", "Online Security", "Device Protection Plan", "Device Protection"],
    "Offering": ["Internet Service", "Internet Type", "Avg Monthly GB Download", "Unlimited Data", "Streaming TV",
                 "Streaming Movies", "Streaming Music", "Multiple Lines", "Online Backup", "Phone Service"],
    "Relationship": ["Contract", "Tenure in Months", "Tenure Months", "Paperless Billing", "Payment Method",
                     "Referred a Friend", "Number of Referrals", "Total Revenue"],
    "Demographic": ["Gender", "Age", "Under 30", "Senior Citizen", "Married", "Partner", "Dependents",
                    "Number of Dependents", "Latitude", "Longitude", "Population"],
}
DMAP, seen = {}, set()
for d, names in DOMAIN_CANDIDATES.items():
    for n in names:
        c = col(n, required=False)
        if c and c not in seen: DMAP.setdefault(d, []).append(c); seen.add(c)
FEATS = [c for fs in DMAP.values() for c in fs]
print("\nFeature -> domain map used:"); [print(f"  {d:12s}: {fs}") for d, fs in DMAP.items()]

# ---------------------------------------------------------------------
# A. Leakage audit
# ---------------------------------------------------------------------
aud = []
for name in ["score", "satisfaction", "cltv"]:
    c = C[name]
    if c is not None:
        v = pd.to_numeric(raw[c], errors="coerce"); m = v.notna()
        aud.append({"Column": c, "Standalone AUC": round(roc_auc_score(y[m], v[m]), 3)})
aud.append({"Column": f"{C['reason']} (non-null)", "Standalone AUC": round(roc_auc_score(y, raw[C["reason"]].notna()), 3)})
if C["status"] is not None:
    aud.append({"Column": f"{C['status']} = Churned", "Standalone AUC": round(roc_auc_score(y, raw[C["status"]].astype(str).str.lower().eq("churned")), 3)})
aud = pd.DataFrame(aud).set_index("Column")
aud["Leakage (|AUC−0.5| > 0.35)"] = (aud["Standalone AUC"] - 0.5).abs() > 0.35
print("\n=== Study 2 leakage audit ==="); save(aud, "S2_T1_leakage")
SAT_OK = C["satisfaction"] is not None and not aud.loc[C["satisfaction"], "Leakage (|AUC−0.5| > 0.35)"]
if SAT_OK:
    DMAP["Satisfaction"] = [C["satisfaction"]]; FEATS.append(C["satisfaction"])
print("Satisfaction Score used as feature:", SAT_OK)

X = raw[FEATS].copy()
NUMF = [c for c in FEATS if pd.api.types.is_numeric_dtype(X[c]) or pd.to_numeric(X[c], errors="coerce").notna().mean() > 0.95]
for c in NUMF: X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
CATF = [c for c in FEATS if c not in NUMF]
for c in CATF: X[c] = X[c].fillna("None").astype(str)
print(f"\n{len(FEATS)} features: {len(NUMF)} numeric, {len(CATF)} categorical")

# ---------------------------------------------------------------------
# B. Risk models, stratified 5-fold, out-of-fold probabilities and SHAP
# ---------------------------------------------------------------------
def pre(): return ColumnTransformer([("num", StandardScaler(), NUMF), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATF)])
def owners(p):
    o = []
    for n, t, cols in p.transformers_:
        if n == "cat": [o.extend([c] * len(cats)) for c, cats in zip(cols, t.categories_)]
        elif n == "num": o.extend(cols)
    return np.array(o)
MODELS = (["CatBoost"] if HAS_CB else []) + ["LogReg"]
SV = {m: np.zeros((len(X), len(FEATS))) for m in MODELS}; PROB = {m: np.zeros(len(X)) for m in MODELS}
skf = StratifiedKFold(K, shuffle=True, random_state=SEED)
for k, (tr, te) in enumerate(skf.split(X, y)):
    for m in MODELS:
        p = pre().fit(X.iloc[tr]); Xtr, Xte = p.transform(X.iloc[tr]), p.transform(X.iloc[te])
        if m == "CatBoost":
            clf = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3, auto_class_weights="Balanced",
                                     random_seed=SEED + k, verbose=0, thread_count=-1).fit(Xtr, y[tr])
            sv = clf.get_feature_importance(Pool(Xte), type="ShapValues")[:, :-1]
        else:
            clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
            sv = clf.coef_[0] * (Xte - Xtr.mean(0))
        SV[m][te] = pd.DataFrame(sv, columns=owners(p)).T.groupby(level=0).sum().T[FEATS].values
        PROB[m][te] = clf.predict_proba(Xte)[:, 1]
perf = pd.DataFrame({m: {"AUC": roc_auc_score(y, PROB[m]), "PR-AUC": average_precision_score(y, PROB[m])} for m in MODELS}).T.round(4)
print("\n=== Study 2 risk models (out-of-fold, 5-fold) ==="); save(perf, "S2_T2_performance")
gi = pd.DataFrame({m: np.abs(SV[m]).mean(0) for m in MODELS}, index=FEATS); gi = gi.div(gi.sum(0), axis=1).sort_values(MODELS[0], ascending=False)
print("\nTop-12 global importance (share of |SHAP|):"); save(gi.head(12).round(4), "S2_T3_global_importance")

# ---------------------------------------------------------------------
# C. Domain-level reason validation (same protocol as Study 1)
# ---------------------------------------------------------------------
GROUPS = {"Competitor": ["Competitor offered higher download speeds", "Competitor offered more data", "Competitor made better offer", "Competitor had better devices"],
          "Price": ["Price too high", "Extra data charges", "Long distance charges", "Lack of affordable download/upload speed"],
          "Support": ["Attitude of support person", "Attitude of service provider", "Poor expertise of phone support", "Poor expertise of online support"],
          "Quality": ["Network reliability", "Product dissatisfaction", "Service dissatisfaction", "Lack of self-service on Website", "Limited range of services"]}
R2G = {r.lower(): g for g, rs in GROUPS.items() for r in rs}
G2D = {"Price": "Price", "Support": "Support", "Competitor": "Offering", "Quality": "Offering"}
reason = raw[C["reason"]].astype(str).str.strip().str.replace("’", "'")
rgroup = reason.str.lower().map(R2G)
val = np.where((y == 1) & rgroup.notna().values)[0]
td = rgroup.iloc[val].map(G2D).values
print(f"\nValidation churners with an actionable stated reason: {len(val)} | unmapped reasons among churners:",
      sorted(set(reason[(y == 1) & rgroup.isna().values]) - {"nan"}))
DOMS = ["Price", "Support", "Offering"]
def shares(att):
    pos = np.clip(att, 0, None); tot = pos.sum(1, keepdims=True); tot[tot == 0] = 1
    return pd.DataFrame({d: pos[:, [FEATS.index(f) for f in fs]].sum(1) / tot[:, 0] for d, fs in DMAP.items()})
rows = []
for m in MODELS:
    sh = shares(SV[m][val]); pred = np.array(DOMS)[sh[DOMS].values.argmax(1)]
    acc = (pred == td).mean(); perm = np.array([(pred == rng.permutation(td)).mean() for _ in range(N_PERM)])
    r = {"Explanation": f"SHAP-{m}", "Alignment acc.": round(acc, 3), "Permutation p": round((np.sum(perm >= acc) + 1) / (N_PERM + 1), 3),
         "Majority baseline": round(pd.Series(td).value_counts(normalize=True).iloc[0], 3), "Cohen κ": round(cohen_kappa_score(td, pred), 3)}
    for d in DOMS:
        lab = (td == d).astype(int); s = sh[d].values
        bs = [roc_auc_score(lab[b], s[b]) for b in (rng.integers(0, len(lab), len(lab)) for _ in range(N_BOOT))]
        r[f"AUC {d} [95% CI]"] = f"{roc_auc_score(lab, s):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"
    r["Non-actionable share"] = round(sh[[d for d in DMAP if d not in DOMS]].sum(1).mean(), 3)
    rows.append(r)
print("\n=== Study 2: explanations vs stated reasons (domain level) ==="); save(pd.DataFrame(rows).set_index("Explanation"), "S2_T4_reason_validation")

# ---------------------------------------------------------------------
# D. Feature-level test for reasons with a directly matching feature
# ---------------------------------------------------------------------
MATCH = {"Extra data charges": ["Total Extra Data Charges"],
         "Long distance charges": ["Total Long Distance Charges", "Avg Monthly Long Distance Charges"],
         "Competitor offered more data": ["Avg Monthly GB Download", "Unlimited Data"],
         "Competitor offered higher download speeds": ["Internet Type", "Avg Monthly GB Download"],
         "Lack of affordable download/upload speed": ["Internet Type", "Monthly Charge", "Monthly Charges"],
         "Price too high": ["Monthly Charge", "Monthly Charges"]}
fl, churn_idx = [], np.where(y == 1)[0]
for rsn, fnames in MATCH.items():
    grp = reason.iloc[churn_idx].str.lower().eq(rsn.lower()).values
    if grp.sum() < 10: continue
    for fn in fnames:
        f = col(fn, required=False)
        if f is None or f not in FEATS: continue
        j = FEATS.index(f)
        if f in NUMF:   # (i) does the feature itself differ for these churners?
            a, b = X[f].values[churn_idx][grp], X[f].values[churn_idx][~grp]
            diff = f"{a.mean():.2f} vs {b.mean():.2f} (MWU p={stats.mannwhitneyu(a, b).pvalue:.3g})"
        else:
            top = X[f].iloc[churn_idx][grp].value_counts(normalize=True).head(1)
            diff = f"{top.index[0]}: {100*top.iloc[0]:.0f}% vs {100*X[f].iloc[churn_idx][~grp].eq(top.index[0]).mean():.0f}%"
        for m in MODELS:   # (ii) does SHAP single out the matching feature for them?
            sv = SV[m][churn_idx]; rank = (np.argsort(np.argsort(-sv, 1), 1)[:, j] + 1)
            auc = roc_auc_score(grp.astype(int), np.clip(sv[:, j], 0, None))
            fl.append({"Stated reason": rsn, "n": int(grp.sum()), "Matching feature": f, "Feature value (reason vs other churners)": diff,
                       "Model": m, "SHAP AUC (feature, reason vs other churners)": round(auc, 3),
                       "Median SHAP rank (reason / others)": f"{np.median(rank[grp]):.0f} / {np.median(rank[~grp]):.0f}"})
print("\n=== Study 2: feature-level test ===")
save(pd.DataFrame(fl), "S2_T5_feature_level", index=False) if fl else print("No directly matching features found.")

# ---------------------------------------------------------------------
# E. Reason recoverability (why-model on churners) with / without geography
# ---------------------------------------------------------------------
yy = LabelEncoder().fit_transform(rgroup.iloc[val].values)
GEO = [c for c in [col("Latitude", required=False), col("Longitude", required=False), col("Population", required=False)] if c]
def why_auc(feats):
    num = [c for c in feats if c in NUMF]; cat = [c for c in feats if c in CATF]
    parts = ([("num", StandardScaler(), num)] if num else []) + ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)] if cat else [])
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(val, yy):
        mdl = Pipeline([("p", ColumnTransformer(parts)), ("c", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                                         class_weight="balanced", random_state=SEED))])
        mdl.fit(X.iloc[val[tr]][feats], yy[tr])
        aucs.append(roc_auc_score(yy[te], mdl.predict_proba(X.iloc[val[te]][feats]), multi_class="ovr", average="macro"))
    return f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
rec = pd.DataFrame([{"Feature set": "All features", "Macro AUC (4 groups)": why_auc(FEATS)},
                    {"Feature set": "Without geography", "Macro AUC (4 groups)": why_auc([f for f in FEATS if f not in GEO])},
                    {"Feature set": "Geography only", "Macro AUC (4 groups)": why_auc(GEO) if GEO else "—"}]).set_index("Feature set")
print("\n=== Study 2: reason recoverability ==="); save(rec, "S2_T6_recoverability")
json.dump({"path": path, "features": FEATS, "domains": DMAP, "satisfaction_used": bool(SAT_OK)}, open(os.path.join(OUT, "study2_config.json"), "w"), indent=1)
print("\nSaved to", OUT)
