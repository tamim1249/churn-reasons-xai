# =====================================================================
# NB9 (single cell): MULTI-DATASET STUDY — when can attribution recover
#                    customers' reasons?  (semi-synthetic ground truth)
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
#
# Public churn datasets have no stated reasons, so we plant them. On each
# real dataset the churn model is trained on REAL churn; only the reason
# labels of churners are simulated, in two worlds:
#   World A "risk-aligned"   : reasons driven by the same features that drive churn risk
#   World B "risk-orthogonal": reasons driven by non-actionable features the risk model barely uses
#                              (mimics Study 1, where reasons follow geography)
# Reason ~ softmax(s * latent + log base-rate), base rates as in IBM (Offering 0.63).
# For each dataset x world x signal s: SHAP alignment vs majority, domain AUC,
# and whether the reasons are recoverable from features at all (why-model).
# Plus a Rashomon check: do equally accurate models agree on the top domain?
#
# Datasets (attach via + Add Input; missing ones are skipped):
#   IBM Telco      : NB1 output in /kaggle/working (telco_clean.csv + config.json)
#   Orange/BigML   : mnassrib/telecom-churn-datasets   (churn-bigml-80/20.csv)
#   Cell2Cell      : jpacse/datasets-for-churn-telecom (cell2celltrain.csv)
#   Iranian (UCI)  : royjafari/customer-churn          (Customer Churn.csv)
# RUNTIME: ~15-25 min on CPU.  Output: /kaggle/working/nb9/
# =====================================================================
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

SEED, N_REP, S_GRID, MAX_N = 42, 20, [0, 0.5, 1, 2, 4], 10000
BASE_RATE = {"Price": 0.15, "Support": 0.22, "Offering": 0.63}
RASHOMON_EPS = 0.02

import os, re, glob, json, itertools, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)
try:
    from catboost import CatBoostClassifier, Pool; HAS_CB = True
except ImportError:
    HAS_CB = False; print("catboost not found -> LogReg models only")

IN_W = os.environ.get("NB_IN", WORK_DIR)
OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb9")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "pdf.fonttype": 42})
def save(df, name, index=True): df.to_csv(os.path.join(TAB, name + ".csv"), index=index); display(df)
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
DOMS = ["Price", "Support", "Offering"]

# ---------------------------------------------------------------------
# 0. Load datasets (detected by their columns, not by folder names)
# ---------------------------------------------------------------------
def read(p): return pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)
def header(p):
    try: return {norm(c) for c in (pd.read_excel(p, nrows=0) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p, nrows=0)).columns}
    except Exception: return set()
files = [p for p in glob.glob(os.environ.get("NB9_GLOB", DATA_DIR + "/**/*"), recursive=True) if p.lower().endswith((".csv", ".xlsx", ".xls"))]
HEAD = {p: header(p) for p in files}
SIGS = {"Orange/BigML": {"customerservicecalls", "totaldaycharge"},
        "Cell2Cell": {"monthlyrevenue", "droppedcalls"},
        "Iranian (UCI)": {"callfailure", "complains"}}
SLUGS = {"Orange/BigML": "mnassrib/telecom-churn-datasets",
         "Cell2Cell": "jpacse/datasets-for-churn-telecom",
         "Iranian (UCI)": "royjafari/customer-churn"}
# Not attached? download automatically (needs Settings -> Internet: On)
for name, sig in SIGS.items():
    if any(sig <= h for h in HEAD.values()): continue
    try:
        import kagglehub
        dl = kagglehub.dataset_download(SLUGS[name]); print(f"{name}: downloaded {SLUGS[name]}")
        for p in glob.glob(os.path.join(dl, "**/*"), recursive=True):
            if p.lower().endswith((".csv", ".xlsx", ".xls")): files.append(p); HEAD[p] = header(p)
    except Exception as e:
        print(f"{name}: auto-download failed ({repr(e)[:120]}) -> attach '{SLUGS[name]}' via + Add Input")
print(f"{len(files)} data files found:"); [print("  ", p) for p in files]
DATA = {}
try:
    cfg = json.load(open(os.path.join(IN_W, "config.json"))); d = pd.read_csv(os.path.join(IN_W, "telco_clean.csv"))
    feats = [f for f in cfg["num_features"] if f != "Avg Monthly Spend"] + cfg["bin_features"] + cfg["cat_features"]
    DATA["IBM Telco"] = (d[feats].copy(), d[cfg["target"]].astype(int).values)
except FileNotFoundError:
    print("IBM Telco skipped (run NB1 first)")
for name, sig in SIGS.items():
    hits = [p for p in files if sig <= HEAD[p] and "holdout" not in p.lower()]
    if not hits: print(f"{name} skipped: no attached file with columns {sig}"); continue
    df = pd.concat([read(p) for p in hits], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]; df = df.drop_duplicates()
    tcol = next(c for c in df.columns if norm(c) in ("churn", "churnvalue", "churnlabel"))
    df = df[df[tcol].notna()]
    yv = df[tcol].astype(str).str.strip().str.lower().isin(["yes", "true", "1", "1.0"]).astype(int).values
    drop = [tcol] + [c for c in df.columns if norm(c) in ("customerid", "phonenumber", "id")]
    X = df.drop(columns=drop).reset_index(drop=True)
    if len(X) > MAX_N:   # stratified subsample for speed
        keep = np.sort(np.concatenate([rng.choice(np.where(yv == k)[0], int(round(MAX_N * (yv == k).mean())), replace=False) for k in (0, 1)]))
        X, yv = X.iloc[keep].reset_index(drop=True), yv[keep]
    DATA[name] = (X, yv)
    print(f"{name}: {len(hits)} file(s) -> {X.shape}")
assert DATA, "No dataset found."

# ---------------------------------------------------------------------
# Feature -> domain (same taxonomy as Study 1; rules listed in the paper's appendix)
# ---------------------------------------------------------------------
IBM_MAP = {**{f: "Price" for f in ["Monthly Charges", "Total Charges", "Charge Gap"]},
           **{f: "Support" for f in ["Tech Support", "Online Security", "Device Protection"]},
           **{f: "Offering" for f in ["Internet Service", "Streaming TV", "Streaming Movies", "Multiple Lines", "Online Backup", "Phone Service", "Num Services"]},
           **{f: "Relationship" for f in ["Contract", "Tenure Months", "Paperless Billing", "Payment Method"]}}
RULES = [("Relationship", r"paperless|payment|contract"),
         ("Support", r"customercare|customerservice|complain|retention|techsupport|support|security|deviceprotection"),
         ("Price", r"charge|revenue|price|amount|overage|tariff|refund"),
         ("Offering", r"minute|calls|called|failure|dropped|blocked|unanswered|usage|frequency|seconds|sms|gb|download|internet|streaming|plan|handset|equipment|roaming|vmail|voicemail|lines|backup|phoneservice|services|webcapable"),
         ("Relationship", r"tenure|length|monthsinservice|subs|referral|credit|status|customervalue|newcellphone")]
def domain_of(c):
    if c in IBM_MAP: return IBM_MAP[c]
    return next((d for d, rx in RULES if re.search(rx, norm(c))), "Other")

def prepare(X):
    X = X.copy(); num, cat = [], []
    for c in X.columns:
        v = pd.to_numeric(X[c], errors="coerce")
        if v.notna().mean() > 0.95 and X[c].nunique() > 2 or pd.api.types.is_numeric_dtype(X[c]):
            X[c] = v.fillna(v.median()); num.append(c)
        elif X[c].nunique() <= 50:
            X[c] = X[c].astype(str).fillna("NA"); cat.append(c)
    return X[num + cat], num, cat

# ---------------------------------------------------------------------
# 1. Risk models (5-fold OOF) + SHAP per original feature
# ---------------------------------------------------------------------
SPECS = ([("CatBoost-d4", "cb", 4), ("CatBoost-d6", "cb", 6), ("CatBoost-d8", "cb", 8)] if HAS_CB else []) + \
        [("LogReg-C1", "lr", 1.0), ("LogReg-C0.1", "lr", 0.1)]
PRIMARY = ["CatBoost-d6", "LogReg-C1"] if HAS_CB else ["LogReg-C1"]
def owners(p):
    o = []
    for n, t, cols in p.transformers_:
        if n == "cat" and len(cols) and hasattr(t, "categories_"): [o.extend([c] * len(cats)) for c, cats in zip(cols, t.categories_)]
        elif n == "num": o.extend(cols)
    return np.array(o)
def fit_all(X, y, num, cat):
    SV = {m: np.zeros((len(X), X.shape[1])) for m, _, _ in SPECS}; PR = {m: np.zeros(len(X)) for m, _, _ in SPECS}
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]).fit(X.iloc[tr])
        Xtr, Xte, own = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te]), owners(pre)
        for m, kind, h in SPECS:
            if kind == "cb":
                clf = CatBoostClassifier(iterations=400, depth=h, learning_rate=0.05, auto_class_weights="Balanced",
                                         random_seed=SEED + k + h, verbose=0, thread_count=-1).fit(Xtr, y[tr])
                sv = clf.get_feature_importance(Pool(Xte), type="ShapValues")[:, :-1]
            else:
                clf = LogisticRegression(C=h, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
                sv = clf.coef_[0] * (Xte - Xtr.mean(0))
            SV[m][te] = pd.DataFrame(sv, columns=own).T.groupby(level=0).sum().T[list(X.columns)].values
            PR[m][te] = clf.predict_proba(Xte)[:, 1]
    return SV, PR

def domain_shares(sv, dom):
    pos = np.clip(sv, 0, None); tot = pos.sum(1, keepdims=True); tot[tot == 0] = 1
    return {d: pos[:, dom == d].sum(1) / tot[:, 0] for d in ["Price", "Support", "Offering", "Relationship", "Other"]}

# ---------------------------------------------------------------------
# 2. Latent reason drivers for the two worlds
# ---------------------------------------------------------------------
def latents(X, y, num, cat, dom_cols, gi_feat):
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]).fit(X)
    Z, own = pre.transform(X), owners(pre)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    sgn = np.sign([np.corrcoef(Z[:, j], y)[0, 1] if Z[:, j].std() > 0 else 0 for j in range(Z.shape[1])])
    dom_own = np.array([dom_cols[o] for o in own])
    A = np.column_stack([(Z[:, dom_own == d] * sgn[dom_own == d]).mean(1) if (dom_own == d).any() else rng.normal(size=len(Z)) for d in DOMS])
    # World B: random projections of NON-ACTIONABLE features (Relationship/Other) that the risk model
    # uses least (bottom half of their global |SHAP|) -> reasons live outside the attribution domains,
    # as geography does in Study 1
    na = gi_feat[[dom_cols[c] not in DOMS for c in gi_feat.index]]
    low = set(na.sort_values().index[: max(1, len(na) // 2)])
    Zl = Z[:, np.isin(own, list(low))]
    B = Zl @ rng.normal(size=(Zl.shape[1], 3)) if Zl.shape[1] else rng.normal(size=(len(Z), 3))
    st = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-9)
    return st(A), st(B), Z, sorted(low)
def draw(L, s):
    logit = s * L + np.log([BASE_RATE[d] for d in DOMS])
    P = np.exp(logit - logit.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    u = rng.random(len(P))[:, None]
    return np.array(DOMS)[(u > P.cumsum(1)).sum(1).clip(0, 2)]
def why_auc(Z, lab):
    yy = pd.Series(lab).map({d: i for i, d in enumerate(DOMS)}).values; aucs = []
    if np.bincount(yy, minlength=3).min() < 5: return np.nan
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Z, yy):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=SEED).fit(Z[tr], yy[tr])
        aucs.append(roc_auc_score(yy[te], m.predict_proba(Z[te]), multi_class="ovr", average="macro"))
    return np.mean(aucs)

# ---------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------
summ, synth, rash = [], [], []
t0 = time.time()
for name, (X0, y) in DATA.items():
    X, num, cat = prepare(X0)
    dom_cols = {c: domain_of(c) for c in X.columns}; dom = np.array([dom_cols[c] for c in X.columns])
    SV, PR = fit_all(X, y, num, cat)
    aucs = {m: roc_auc_score(y, PR[m]) for m in SV}
    gi = pd.Series(np.abs(SV[PRIMARY[0]]).mean(0), index=X.columns)
    gshare = gi.groupby(dom).sum() / gi.sum()
    summ.append({"Dataset": name, "n": len(X), "Churn rate": round(y.mean(), 3), "Features": X.shape[1],
                 **{f"# {d}": int((dom == d).sum()) for d in DOMS + ["Relationship", "Other"]},
                 **{f"AUC {m}": round(aucs[m], 3) for m in PRIMARY},
                 **{f"|SHAP| share {d}": round(gshare.get(d, 0), 3) for d in DOMS}})
    ch = np.where(y == 1)[0]
    A, B, Z, low = latents(X, y, num, cat, dom_cols, gi)
    A, B, Zc = A[ch], B[ch], Z[ch]
    shares = {m: domain_shares(SV[m][ch], dom) for m in PRIMARY}
    ind = {m: np.array(DOMS)[np.column_stack([shares[m][d] for d in DOMS]).argmax(1)] for m in PRIMARY}
    for world, L in [("A risk-aligned", A), ("B risk-orthogonal", B)]:
        for s in S_GRID:
            rec = {m: {"gain": [], "auc": []} for m in PRIMARY}; why = []
            for r in range(N_REP):
                lab = draw(L, s); maj = pd.Series(lab).value_counts(normalize=True).iloc[0]
                for m in PRIMARY:
                    rec[m]["gain"].append((ind[m] == lab).mean() - maj)
                    rec[m]["auc"].append(np.nanmean([roc_auc_score(lab == d, shares[m][d]) for d in DOMS if 0 < (lab == d).mean() < 1]))
                if s == 2 and r < 3: why.append(why_auc(Zc, lab))
            for m in PRIMARY:
                synth.append({"Dataset": name, "World": world, "s": s, "Model": m,
                              "Accuracy gain vs majority": np.mean(rec[m]["gain"]), "Gain SD": np.std(rec[m]["gain"]),
                              "Mean domain AUC": np.mean(rec[m]["auc"]), "AUC SD": np.std(rec[m]["auc"]),
                              "Why-model AUC (s=2)": np.nanmean(why) if why else np.nan})
    # Rashomon: models within EPS of the best AUC; agreement on churners' top actionable domain
    good = [m for m in SV if aucs[m] >= max(aucs.values()) - RASHOMON_EPS]
    tops = {m: np.array(DOMS)[np.column_stack([domain_shares(SV[m][ch], dom)[d] for d in DOMS]).argmax(1)] for m in good}
    pair = [(tops[a] == tops[b]).mean() for a, b in itertools.combinations(good, 2)]
    allag = np.mean([len({tops[m][i] for m in good}) == 1 for i in range(len(ch))]) if len(good) > 1 else np.nan
    rash.append({"Dataset": name, "Models in Rashomon set (ε=0.02)": ", ".join(good),
                 "AUC range": f"{min(aucs[m] for m in good):.3f}–{max(aucs[m] for m in good):.3f}",
                 "Mean pairwise top-domain agreement (%)": round(100 * np.mean(pair), 1) if pair else np.nan,
                 "All models agree (%)": round(100 * allag, 1) if pair else np.nan})
    print(f"{name}: done | {(time.time()-t0)/60:.1f} min | World-B drivers: {low[:6]}{' ...' if len(low) > 6 else ''}")

print("\n=== Datasets and risk models ==="); save(pd.DataFrame(summ).set_index("Dataset"), "T26_datasets")
syn = pd.DataFrame(synth)
syn.round(3).to_csv(os.path.join(TAB, "T27_semisynthetic_full.csv"), index=False)
key = syn[syn.s.isin([0, 2])].copy()
key["Gain [SD]"] = key.apply(lambda r: f"{r['Accuracy gain vs majority']:+.3f} [{r['Gain SD']:.3f}]", axis=1)
key["Domain AUC [SD]"] = key.apply(lambda r: f"{r['Mean domain AUC']:.3f} [{r['AUC SD']:.3f}]", axis=1)
key = key.pivot_table(index=["Dataset", "World", "Model"], columns="s", values=["Gain [SD]", "Domain AUC [SD]"], aggfunc="first")
key.columns = [f"{a} (s={b})" for a, b in key.columns]
key = key.join(syn[syn.s == 2].groupby(["Dataset", "World", "Model"])["Why-model AUC (s=2)"].first().round(3))
print("\n=== Semi-synthetic reason recovery (s=0: no signal, s=2: strong signal) ==="); save(key, "T28_semisynthetic_key")
print("\n=== Rashomon agreement on churners' top actionable domain ==="); save(pd.DataFrame(rash).set_index("Dataset"), "T29_rashomon")

# ---------------------------------------------------------------------
# Figure 10: domain AUC vs reason signal strength, per dataset
# ---------------------------------------------------------------------
dsn = list(DATA); fig, axs = plt.subplots(1, len(dsn), figsize=(2.6 * len(dsn) + 0.6, 2.8), sharey=True, squeeze=False)
col = {"A risk-aligned": "#3B5B92", "B risk-orthogonal": "#C0392B"}
for ax, dn in zip(axs[0], dsn):
    for (w, m), g in syn[syn.Dataset == dn].groupby(["World", "Model"]):
        ls = "-" if m == PRIMARY[0] else "--"
        ax.errorbar(g.s, g["Mean domain AUC"], yerr=g["AUC SD"], color=col[w], ls=ls, marker="o", ms=3, capsize=2,
                    label=f"{w.split()[0]}: {w.split(' ', 1)[1]} · {m}")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":"); ax.set_title(dn); ax.set_xlabel("Reason signal strength s")
axs[0][0].set_ylabel("Mean domain AUC (SHAP share)")
h, l = axs[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
plt.tight_layout()
for ext in ["png", "pdf"]: plt.savefig(os.path.join(FIG, f"Fig_10_semisynthetic.{ext}"), bbox_inches="tight")
plt.show()
print("\nSaved to", OUT, "->", sorted(os.listdir(TAB)))
