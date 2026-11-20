# =====================================================================
# NB2 (single cell): Stage-1 Churn Prediction with Nested Repeated CV
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
# Solves: L3 (no nested CV / no stat tests), L4 (SMOTE leakage),
#         L9 (calibration ignored)
# Input : NB1 output (telco_clean.csv, config.json)
# Output: /kaggle/working/nb2/  -> fold metrics, OOF predictions,
#         best params, stat tests, figures (300 dpi)
#
# RUNTIME: see MODE below (balanced ~45-70 min on Kaggle CPU). Use "Save Version ->
#          Save & Run All (Commit)" so it runs in the background.
#          MODE = "fast" first to test everything in ~10 min.
# =====================================================================

# ---------------------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------------------
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

MODE = os.environ.get("MODE", "balanced")  # "fast" (~10 min test) | "balanced" (~45-70 min) | "full" (~2-3 h)

N_SPLITS   = 5                                                   # outer folds
N_REPEATS  = {"fast": 1, "balanced": 2, "full": 3}[MODE]         # 5 x 2 = 10 outer folds
INNER_K    = 3                                                   # inner folds for tuning
N_TRIALS   = {"fast": 5, "balanced": 15, "full": 25}[MODE]       # Optuna trials per outer fold
FAST       = MODE == "fast"
RESUME     = True     # True = already-finished models are loaded from disk, not retrained
SEED       = 42

import os, json, time, warnings, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score, brier_score_loss, log_loss,
                             roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Optuna nai. Kaggle e sadharonoto thake; na thakle: !pip install optuna")

AVAILABLE = {}
try:
    from xgboost import XGBClassifier; AVAILABLE["XGBoost"] = True
except ImportError: print("xgboost not found -> skipped")
try:
    from lightgbm import LGBMClassifier; AVAILABLE["LightGBM"] = True
except ImportError: print("lightgbm not found -> skipped")
try:
    from catboost import CatBoostClassifier; AVAILABLE["CatBoost"] = True
except ImportError: print("catboost not found -> skipped")
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE; HAS_IMB = True
except ImportError:
    HAS_IMB = False; print("imblearn not found -> SMOTE ablation skipped")

# ---------------------------------------------------------------------
# 1. Paths + load NB1 output
# ---------------------------------------------------------------------
def find_file(name):
    """NB1 output: same notebook (/kaggle/working) or attached as input."""
    for p in [os.path.join(os.environ.get("NB_IN", WORK_DIR), name)] + \
             glob.glob(f"{DATA_DIR}/**/{name}", recursive=True):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} pai nai. Age NB1 run koro, ba NB1 er output notebook ta input hishebe attach koro.")

OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb2")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)

cfg  = json.load(open(find_file("config.json")))
data = pd.read_csv(find_file("telco_clean.csv"))
TARGET = cfg["target"]

# Drop near-duplicate of Monthly Charges (would split SHAP credit, hurts reason validation)
DROP_FEATS = ["Avg Monthly Spend"]
NUM  = [f for f in cfg["num_features"] if f not in DROP_FEATS]
BIN  = cfg["bin_features"]
CAT  = cfg["cat_features"]
FEATS = NUM + BIN + CAT
assert not set(FEATS) & set(cfg["leakage_cols"] + [cfg["value_col"], TARGET])

X = data[FEATS].copy()
y = data[TARGET].astype(int).values
POS_W = (y == 0).sum() / (y == 1).sum()
print(f"Data: {X.shape} | churn rate {y.mean():.4f} | pos weight {POS_W:.2f}")
print(f"Outer CV: {N_SPLITS}x{N_REPEATS} = {N_SPLITS*N_REPEATS} folds | inner {INNER_K}-fold | {N_TRIALS} trials")

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight"); plt.show()

# ---------------------------------------------------------------------
# 2. Preprocessing (fit INSIDE each fold via Pipeline -> no leakage)
# ---------------------------------------------------------------------
def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("bin", "passthrough", BIN),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])

# ---------------------------------------------------------------------
# 3. Models + Optuna search spaces
#    Imbalance handled by class weights (SMOTE tested separately as ablation)
# ---------------------------------------------------------------------
def build(name, p, seed):
    if name == "LogReg":
        return LogisticRegression(C=p["C"], class_weight="balanced", max_iter=3000)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                      class_weight="balanced", n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                             learning_rate=p["learning_rate"], subsample=p["subsample"],
                             colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
                             reg_lambda=p["reg_lambda"], scale_pos_weight=POS_W,
                             eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=p["n_estimators"], num_leaves=p["num_leaves"],
                              learning_rate=p["learning_rate"], min_child_samples=p["min_child_samples"],
                              subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                              reg_lambda=p["reg_lambda"], class_weight="balanced",
                              n_jobs=-1, random_state=seed, verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"],
                                  l2_leaf_reg=p["l2_leaf_reg"], auto_class_weights="Balanced",
                                  random_seed=seed, verbose=0, thread_count=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"], learning_rate_init=p["lr"],
                             early_stopping=True, max_iter=500, random_state=seed)

MLP_ARCH = {"32": (32,), "64": (64,), "64-32": (64, 32)}

def suggest(name, t):
    if name == "LogReg":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if name == "RandomForest":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 600, step=100),
                "max_depth": t.suggest_int("max_depth", 3, 16),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
                "max_features": t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5])}
    if name == "XGBoost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": t.suggest_int("max_depth", 2, 6),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "LightGBM":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
                "num_leaves": t.suggest_int("num_leaves", 8, 64),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 10, 100),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "CatBoost":
        return {"iterations": t.suggest_int("iterations", 200, 600, step=100),
                "depth": t.suggest_int("depth", 3, 7),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True)}
    if name == "MLP":
        return {"hidden": MLP_ARCH[t.suggest_categorical("hidden", list(MLP_ARCH))],
                "alpha": t.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True)}

MODELS = ["LogReg", "RandomForest"] + [m for m in ["XGBoost", "LightGBM", "CatBoost"] if m in AVAILABLE] + ["MLP"]
print("Models:", MODELS)

def pipe(name, p, seed):
    return Pipeline([("pre", make_pre()), ("clf", build(name, p, seed))])

def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"AUC": roc_auc_score(y_true, prob), "PR_AUC": average_precision_score(y_true, prob),
            "F1": f1_score(y_true, pred), "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred), "BalAcc": balanced_accuracy_score(y_true, pred),
            "Brier": brier_score_loss(y_true, prob), "LogLoss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7))}

# ---------------------------------------------------------------------
# 4. Nested repeated CV (outer = evaluation, inner = Optuna tuning)
#    Same outer splits for every model -> paired statistical tests
# ---------------------------------------------------------------------
outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
SPLITS = list(outer.split(X, y))
json.dump([{"fold": i, "repeat": i // N_SPLITS, "test_idx": te.tolist()} for i, (_, te) in enumerate(SPLITS)],
          open(os.path.join(OUT, "outer_splits.json"), "w"))

fold_rows, oof_rows, param_rows = [], [], []
BEST = {}   # (model, fold) -> Optuna best_params

INT_P = {"n_estimators", "max_depth", "min_samples_leaf", "num_leaves", "min_child_samples",
         "min_child_weight", "iterations", "depth"}
def clean_params(d):
    """Restore Optuna params read back from CSV (NaN removal + type fixes)."""
    out = {}
    for kk, v in d.items():
        if kk in ("Model", "Fold") or (isinstance(v, float) and np.isnan(v)):
            continue
        if kk in INT_P: v = int(v)
        elif kk == "max_features": v = v if str(v) == "sqrt" else float(v)
        elif kk == "hidden": v = str(int(v)) if isinstance(v, (int, float)) else str(v)
        else: v = float(v)
        out[kk] = v
    return out

DONE = set()
paths = {n: os.path.join(OUT, f) for n, f in [("fm", "fold_metrics.csv"), ("oof", "oof_predictions.csv"), ("bp", "best_params.csv")]}
if RESUME and all(os.path.exists(p) for p in paths.values()):
    prev_fm, prev_oof, prev_bp = (pd.read_csv(paths[n]) for n in ["fm", "oof", "bp"])
    same_cv = prev_fm["Fold"].nunique() == len(SPLITS) and \
              set(prev_oof.loc[(prev_oof.Model == prev_oof.Model.iloc[0]) & (prev_oof.Fold == 0), "row"]) == set(SPLITS[0][1].tolist())
    if same_cv:
        DONE = {m for m in MODELS if (prev_fm.Model == m).sum() == len(SPLITS) and (prev_bp.Model == m).sum() == len(SPLITS)}
        fold_rows  = prev_fm[prev_fm.Model.isin(DONE)].to_dict("records")
        oof_rows   = [prev_oof[prev_oof.Model.isin(DONE)]]
        param_rows = prev_bp[prev_bp.Model.isin(DONE)].to_dict("records")
        for r in param_rows:
            BEST[(r["Model"], int(r["Fold"]))] = clean_params(r)
        print("RESUME: loaded finished models ->", sorted(DONE))
    else:
        print("RESUME: saved results use a different CV setup -> retraining everything")

t_all = time.time()
for name in MODELS:
    if name in DONE:
        print(f"{name:13s} skipped (loaded from disk)")
        continue
    t0 = time.time()
    for k, (tr, te) in enumerate(SPLITS):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        inner = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=SEED + k)

        def objective(trial):
            p = suggest(name, trial)
            cv_jobs = INNER_K if name in ["LogReg", "MLP"] else 1   # LogReg/MLP single-threaded -> run folds in parallel
            return cross_val_score(pipe(name, p, SEED), Xtr, ytr, cv=inner, scoring="roc_auc", n_jobs=cv_jobs).mean()

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + k))
        study.optimize(objective, n_trials=1 if name == "LogReg" and FAST else N_TRIALS, show_progress_bar=False)
        best = suggest(name, optuna.trial.FixedTrial(study.best_params))

        model = pipe(name, best, SEED + k).fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]

        print(f"  {name} fold {k+1}/{len(SPLITS)}: AUC {roc_auc_score(yte, prob):.4f} "
              f"| elapsed {(time.time()-t0)/60:.1f} min", flush=True)
        m = metrics(yte, prob); m.update({"Model": name, "Fold": k, "Repeat": k // N_SPLITS, "InnerAUC": study.best_value})
        fold_rows.append(m)
        param_rows.append({"Model": name, "Fold": k, **study.best_params})
        BEST[(name, k)] = study.best_params
        oof_rows.append(pd.DataFrame({"Model": name, "Fold": k, "Repeat": k // N_SPLITS,
                                      "row": te, "y": yte, "prob": prob}))
    folds_df = pd.DataFrame(fold_rows)
    print(f"{name:13s} AUC {folds_df.loc[folds_df.Model==name,'AUC'].mean():.4f} "
          f"± {folds_df.loc[folds_df.Model==name,'AUC'].std():.4f} | {(time.time()-t0)/60:.1f} min")
    # save after each model (Kaggle session crash hole progress thakbe)
    folds_df.to_csv(os.path.join(OUT, "fold_metrics.csv"), index=False)
    pd.concat(oof_rows).to_csv(os.path.join(OUT, "oof_predictions.csv"), index=False)
    pd.DataFrame(param_rows).to_csv(os.path.join(OUT, "best_params.csv"), index=False)
print(f"Total: {(time.time()-t_all)/60:.1f} min")

folds_df = pd.DataFrame(fold_rows)
oof = pd.concat(oof_rows, ignore_index=True)

# ---------------------------------------------------------------------
# 5. Table: performance (mean ± sd over outer folds)
# ---------------------------------------------------------------------
MET = ["AUC", "PR_AUC", "F1", "Precision", "Recall", "BalAcc", "Brier", "LogLoss"]
agg = folds_df.groupby("Model")[MET].agg(["mean", "std"])
perf = pd.DataFrame({m: agg[(m, "mean")].map("{:.4f}".format) + " ± " + agg[(m, "std")].map("{:.4f}".format) for m in MET})
perf = perf.loc[agg[("AUC", "mean")].sort_values(ascending=False).index]
perf.to_csv(os.path.join(TAB, "T2_performance.csv"))
print("\n=== Table 2: Outer-fold performance (mean ± sd) ==="); display(perf)

# ---------------------------------------------------------------------
# 6. Statistical tests
#    (a) Friedman on per-fold AUC ranks + Nemenyi critical difference
#    (b) Nadeau-Bengio corrected resampled t-test vs best model, Holm-adjusted
# ---------------------------------------------------------------------
auc_mat = folds_df.pivot(index="Fold", columns="Model", values="AUC")[list(perf.index)]
k_mod, n_blk = auc_mat.shape[1], auc_mat.shape[0]
fr_stat, fr_p = (stats.friedmanchisquare(*[auc_mat[c].values for c in auc_mat.columns])
                 if k_mod >= 3 else (np.nan, np.nan))
ranks = auc_mat.rank(axis=1, ascending=False).mean().sort_values()
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}   # Demsar (2006)
CD = Q05[k_mod] * np.sqrt(k_mod * (k_mod + 1) / (6 * n_blk))
print(f"\nFriedman chi2 = {fr_stat:.3f}, p = {fr_p:.2e} | Nemenyi CD (alpha=.05) = {CD:.3f}")

n_te = len(SPLITS[0][1]); n_tr = len(SPLITS[0][0]); J = n_blk
best_model = perf.index[0]
rows = []
for mdl in auc_mat.columns:
    if mdl == best_model: continue
    d = auc_mat[best_model].values - auc_mat[mdl].values
    var_c = (1 / J + n_te / n_tr) * d.var(ddof=1)
    t = d.mean() / np.sqrt(var_c) if var_c > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    rows.append({"Comparison": f"{best_model} vs {mdl}", "Mean ΔAUC": d.mean(), "t": t, "p_raw": p})
tt = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
m_ = len(tt)
tt["p_holm"] = np.maximum.accumulate(np.minimum(1, tt["p_raw"] * (m_ - np.arange(m_))))
tt["Significant (0.05)"] = tt["p_holm"] < 0.05
stat_tab = pd.concat([ranks.rename("Mean rank").to_frame()], axis=1)
stat_tab.to_csv(os.path.join(TAB, "T3_mean_ranks.csv"))
tt.round(5).to_csv(os.path.join(TAB, "T3_corrected_ttest.csv"), index=False)
print("Mean ranks:"); display(ranks.round(3).to_frame("Mean rank"))
print("Corrected resampled t-test (Holm):"); display(tt.round(5))

# ---------------------------------------------------------------------
# 7. SMOTE ablation (SMOTE strictly INSIDE training fold) — L4
#    Uses first repeat's folds and each fold's tuned params; compares
#    class-weight vs SMOTE for LogReg and best boosting model
# ---------------------------------------------------------------------
if HAS_IMB:
    abl = []
    boost = [m for m in perf.index if m in ["XGBoost", "LightGBM", "CatBoost"]]
    abl_models = ["LogReg"] + boost[:1]
    for name in abl_models:
        for k in range(N_SPLITS):
            tr, te = SPLITS[k]
            p = suggest(name, optuna.trial.FixedTrial(BEST[(name, k)]))
            clf = build(name, p, SEED + k)
            # turn off class weighting for the SMOTE variant
            if name == "LogReg": clf.set_params(class_weight=None)
            elif name == "XGBoost": clf.set_params(scale_pos_weight=1)
            elif name == "LightGBM": clf.set_params(class_weight=None)
            elif name == "CatBoost":   # auto_class_weights=None is not accepted -> rebuild without it
                clf = CatBoostClassifier(iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"],
                                         l2_leaf_reg=p["l2_leaf_reg"], random_seed=SEED + k, verbose=0, thread_count=-1)
            sm = ImbPipeline([("pre", make_pre()), ("smote", SMOTE(random_state=SEED + k)), ("clf", clf)])
            prob = sm.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            r = metrics(y[te], prob); r.update({"Model": name, "Imbalance": "SMOTE (in-fold)", "Fold": k}); abl.append(r)
            r2 = folds_df[(folds_df.Model == name) & (folds_df.Fold == k)][MET].iloc[0].to_dict()
            r2.update({"Model": name, "Imbalance": "Class weight", "Fold": k}); abl.append(r2)
    abl = pd.DataFrame(abl).groupby(["Model", "Imbalance"])[["AUC", "PR_AUC", "F1", "Recall", "Brier"]].mean().round(4)
    abl.to_csv(os.path.join(TAB, "T4_smote_ablation.csv"))
    print("\n=== SMOTE (in-fold) vs class weights, repeat 1 ==="); display(abl)

# ---------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------
order = list(perf.index)
cols = plt.cm.tab10(np.linspace(0, 1, 10))[:len(order)]
CMAP = dict(zip(order, cols))

# F7: AUC distribution over outer folds
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([auc_mat[m] for m in order], vert=False, patch_artist=True,
           boxprops=dict(facecolor="#DCE6F2"), medianprops=dict(color="#C8553D"))
ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order); ax.invert_yaxis(); ax.set_xlabel("AUC (outer folds)")
ax.set_title(f"AUC across {n_blk} outer folds ({N_SPLITS}x{N_REPEATS} nested CV)")
savefig("F7_auc_boxplot")

# F8: Critical difference style plot
fig, ax = plt.subplots(figsize=(8, 2.4))
ax.hlines(0, 1, k_mod, color="k", lw=1)
for i, (mdl, r) in enumerate(ranks.items()):
    ax.plot(r, 0, "o", color=CMAP[mdl]); ax.annotate(f"{mdl} ({r:.2f})", (r, 0), xytext=(r, 0.35 + 0.25 * (i % 3)),
                                                    ha="center", fontsize=9, arrowprops=dict(arrowstyle="-", lw=0.6))
ax.plot([1, 1 + CD], [-0.4, -0.4], color="#C8553D", lw=3); ax.text(1 + CD / 2, -0.7, f"CD = {CD:.2f}", ha="center", fontsize=9)
ax.set_xlim(0.8, k_mod + 0.2); ax.set_ylim(-1, 1.3); ax.set_yticks([]); ax.set_xlabel("Mean rank (lower = better)")
ax.spines["left"].set_visible(False)
ax.set_title(f"Friedman p = {fr_p:.1e}; models closer than CD are not significantly different")
savefig("F8_critical_difference")

# F9: ROC + PR curves (repeat 0 OOF = every customer predicted once)
o0 = oof[oof.Repeat == 0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for mdl in order:
    s = o0[o0.Model == mdl]
    fpr, tpr, _ = roc_curve(s.y, s.prob); pr, rc, _ = precision_recall_curve(s.y, s.prob)
    axes[0].plot(fpr, tpr, color=CMAP[mdl], lw=1.4, label=f"{mdl} ({roc_auc_score(s.y, s.prob):.3f})")
    axes[1].plot(rc, pr, color=CMAP[mdl], lw=1.4, label=f"{mdl} ({average_precision_score(s.y, s.prob):.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
axes[1].axhline(y.mean(), ls="--", color="grey", lw=0.8)
axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (out-of-fold, repeat 1)")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (out-of-fold, repeat 1)")
for a in axes: a.legend(frameon=False, fontsize=8)
savefig("F9_roc_pr")

# F10: Calibration (important for profit layer)
fig, ax = plt.subplots(figsize=(5.5, 5))
for mdl in order:
    s = o0[o0.Model == mdl]
    fp, mp = calibration_curve(s.y, s.prob, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", ms=3, lw=1.2, color=CMAP[mdl], label=mdl)
ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration (out-of-fold)")
ax.legend(frameon=False, fontsize=8)
savefig("F10_calibration")

print("\nNote: class-weighted models are often over-confident upward (curve below diagonal).")
print("NB5 (profit) e isotonic calibration apply kora hobe inner-fold e.")
print("\nSaved to", OUT, "->", sorted(os.listdir(OUT)))
