# =====================================================================
# NB5 (single cell): Profit evaluation of retention strategies
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
#
# RQ4: Does choosing the retention offer from SHAP (as prior work does)
#      pay off, compared with a dedicated why-model and with an oracle?
# Solves: L6 (profit metrics ignore reasons / uncontrollable churn),
#         L7 (prescriptions never checked against the real cause),
#         L9 (calibration before value-based targeting)
#
# Customer-level realised profit on OUT-OF-FOLD predictions (repeat 1):
#   targeted churner   : gamma(offer, true reason) * CLTV - cost
#   targeted non-churner: - cost
#   Baseline offer = most common reason domain in training folds ("majority offer")
#   gamma = G_MATCH if offer matches true reason domain, G_MISS otherwise,
#           0 for Uncontrollable (Moved/Deceased), random-offer average for Unknown
# EMP: expected maximum profit with gamma_match ~ Beta (Monte-Carlo),
#      in the spirit of EMPC (Verbeke et al. 2012) with customer-level CLTV
#      (as in e-Profits) — not an exact re-implementation of either.
#
# Input : NB1-NB3 output in /kaggle/working   RUNTIME: ~3-6 min
# =====================================================================
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

G_MATCH, G_MISS = 0.40, 0.10        # offer success if matched / mismatched to true reason
COST            = 100               # contact + incentive cost per targeted customer
N_BOOT          = 1000
SEED            = 42

import os, json, glob, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

BASE = os.environ.get("NB_IN", WORK_DIR)
def find_file(name):
    for p in [os.path.join(BASE, name)] + glob.glob(f"{DATA_DIR}/**/{name}", recursive=True):
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
OUT = os.path.join(os.environ.get("NB_OUT", WORK_DIR), "nb5")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in [OUT, FIG, TAB]: os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})
def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight"); plt.show()

cfg    = json.load(open(find_file("config.json")))
data   = pd.read_csv(find_file("telco_clean.csv"))
splits = [s for s in json.load(open(find_file("nb2/outer_splits.json"))) if s["repeat"] == 0]
NUM = [f for f in cfg["num_features"] if f != "Avg Monthly Spend"]
BIN, CAT = cfg["bin_features"], cfg["cat_features"]
FEATS = NUM + BIN + CAT
y    = data[cfg["target"]].astype(int).values
CLTV = data[cfg["value_col"]].astype(float).values
N    = len(data)
fold_of = np.empty(N, int)
for s in splits: fold_of[s["test_idx"]] = s["fold"]

DOMAINS = ["Price", "Support", "Offering"]
OFFER = {"Price": "Discount / price lock", "Support": "Priority support call", "Offering": "Plan upgrade / speed boost"}
R2D = {"Price": "Price", "Support": "Support", "Competitor": "Offering", "Quality": "Offering"}
rg = data["Reason Group"].astype(object).values
true_dom = np.array([R2D.get(r, None) if isinstance(r, str) else None for r in rg], dtype=object)
uncontrollable = (rg == "Uncontrollable")
unknown        = (rg == "Unknown")

# ---------------------------------------------------------------------
# 1. Churn probabilities (out-of-fold, NB3) + cross-fitted isotonic calibration
# ---------------------------------------------------------------------
PROB = {}
for m in ["CatBoost", "XGBoost", "LogReg"]:
    try: PROB[m] = pd.read_csv(find_file(f"nb3/shap_{m}.csv"))["prob"].values
    except FileNotFoundError: pass
PRIMARY = next(iter(PROB))
print("Churn models:", list(PROB), "| primary:", PRIMARY)

def crossfit_isotonic(p):
    out = np.zeros_like(p)
    for k in np.unique(fold_of):
        tr, te = fold_of != k, fold_of == k        # calibrator never sees fold k labels
        out[te] = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr]).predict(p[te])
    return out
PCAL = {m: crossfit_isotonic(p) for m, p in PROB.items()}
cal_tab = pd.DataFrame({m: {"AUC raw": roc_auc_score(y, PROB[m]), "Brier raw": brier_score_loss(y, PROB[m]),
                            "Brier calibrated": brier_score_loss(y, PCAL[m]),
                            "Mean p raw": PROB[m].mean(), "Mean p calibrated": PCAL[m].mean(), "Churn rate": y.mean()}
                        for m in PROB}).T
cal_tab.round(4).to_csv(os.path.join(TAB, "T15_calibration.csv"))
print("\n=== Table 15: Calibration (cross-fitted isotonic) ==="); display(cal_tab.round(4))

# ---------------------------------------------------------------------
# 2. Offer-choice rules (all out-of-fold)
#    a) SHAP-guided: domain with largest positive SHAP share (what prior work implies)
#    b) Why-model: trained on churners' stated reasons in training folds
#    c) Oracle: true stated reason (upper bound)     d) Random offer (baseline)
# ---------------------------------------------------------------------
DMAP = {"Price": ["Monthly Charges", "Total Charges", "Charge Gap"],
        "Support": ["Tech Support", "Online Security", "Device Protection"],
        "Offering": ["Internet Service", "Streaming TV", "Streaming Movies", "Multiple Lines",
                     "Online Backup", "Phone Service", "Num Services"]}
sv = pd.read_csv(find_file(f"nb3/shap_{PRIMARY}.csv"))[FEATS].values
pos = np.clip(sv, 0, None)
shap_dom = np.array(DOMAINS)[np.column_stack([pos[:, [FEATS.index(f) for f in DMAP[d]]].sum(1) for d in DOMAINS]).argmax(1)]

why_dom = np.empty(N, dtype=object)
act = np.array([t is not None for t in true_dom]) & (y == 1)
for k in np.unique(fold_of):
    tr = (fold_of != k) & act; te = fold_of == k
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("bin", "passthrough", BIN),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    # no class balancing: the profit-optimal offer is the MOST PROBABLE reason (maximises expected gamma)
    why = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                         random_state=SEED))])
    why.fit(data.loc[tr, FEATS], true_dom[tr].astype(str))
    why_dom[te] = why.predict(data.loc[te, FEATS])
rand_dom = rng.choice(DOMAINS, N)
# Majority offer: most common stated-reason domain among TRAINING-fold churners (strong naive baseline)
maj_dom = np.empty(N, dtype=object)
for k in np.unique(fold_of):
    tr = (fold_of != k) & act
    maj_dom[fold_of == k] = pd.Series(true_dom[tr]).value_counts().index[0]

acc_tab = pd.DataFrame({name: {"Offer matches stated reason (actionable churners)": (dom[act] == true_dom[act]).mean()}
                        for name, dom in [("SHAP-guided", shap_dom), ("Why-model", why_dom),
                                          ("Majority offer", maj_dom), ("Random", rand_dom)]}).T
acc_tab.loc["Oracle"] = 1.0
print("\nOffer-reason match rate:"); display(acc_tab.round(3))

# ---------------------------------------------------------------------
# 3. Profit engine
# ---------------------------------------------------------------------
def gamma_vec(offer_dom, g_match=G_MATCH, g_miss=G_MISS):
    g = np.where(offer_dom == true_dom, g_match, g_miss).astype(float)
    g[unknown] = (g_match + 2 * g_miss) / 3            # unknown cause: expected value of a random offer
    g[uncontrollable] = 0.0                             # moved / deceased cannot be retained
    g[y == 0] = 0.0                                     # non-churners: nothing to save
    return g

def profit_curve(score, offer_dom, g_match=G_MATCH, g_miss=G_MISS, cost=COST, idx=None):
    idx = np.arange(N) if idx is None else idx
    gain = gamma_vec(offer_dom, g_match, g_miss)[idx] * CLTV[idx] - cost
    order = np.argsort(-score[idx], kind="stable")
    return np.concatenate([[0], np.cumsum(gain[order])])   # profit when targeting top-k

BASELINE = "S2 Risk×CLTV + majority offer"
STRATS = {   # name: (ranking score, offer rule)
    "S1 Risk rank + majority offer":        (PCAL[PRIMARY],        maj_dom),
    BASELINE:                               (PCAL[PRIMARY] * CLTV, maj_dom),
    "S2r Risk×CLTV + random offer":         (PCAL[PRIMARY] * CLTV, rand_dom),
    "S3 Risk×CLTV + SHAP-guided offer":     (PCAL[PRIMARY] * CLTV, shap_dom),
    "S4 Risk×CLTV + why-model offer":       (PCAL[PRIMARY] * CLTV, why_dom),
    "S5 Risk×CLTV + oracle offer (bound)":  (PCAL[PRIMARY] * CLTV, true_dom),
    "S2u Risk×CLTV uncalibrated + majority":(PROB[PRIMARY] * CLTV, maj_dom),
}
curves = {s: profit_curve(sc, od) for s, (sc, od) in STRATS.items()}

def emp(score, offer_dom, n_mc=300):
    # EMP: average over gamma_match ~ Beta(6,14) (mean 0.30), gamma_miss = gamma_match/4
    vals = []
    for g in rng.beta(6, 14, n_mc):
        vals.append(profit_curve(score, offer_dom, g, g / 4).max())
    return np.mean(vals) / N

rows = []
for s, (sc, od) in STRATS.items():
    c = curves[s]; k = c.argmax()
    rows.append({"Strategy": s, "Max profit": c.max(), "Optimal % targeted": 100 * k / N,
                 "Profit per customer": c.max() / N, "EMP per customer": emp(sc, od)})
prof = pd.DataFrame(rows).set_index("Strategy")

# bootstrap CI for differences vs S2 (paired, same resampled customers)
def boot_diff(a, b):
    sa, oa = STRATS[a]; sb, ob = STRATS[b]; diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, N, N)
        diffs.append(profit_curve(sa, oa, idx=i).max() - profit_curve(sb, ob, idx=i).max())
    return np.mean(diffs), *np.percentile(diffs, [2.5, 97.5])
for s in prof.index:
    if s == BASELINE: continue
    m_, lo, hi = boot_diff(s, BASELINE)
    prof.loc[s, "Δ vs S2 [95% CI]"] = f"{m_:,.0f} [{lo:,.0f}, {hi:,.0f}]"
prof.round(2).to_csv(os.path.join(TAB, "T16_profit_strategies.csv"))
print(f"\n=== Table 16: Retention profit (out-of-fold, {PRIMARY}, γ_match={G_MATCH}, γ_miss={G_MISS}, cost={COST}) ===")
display(prof.round(2))

# model comparison under the same strategy
mrows = []
for m in PROB:
    c = profit_curve(PCAL[m] * CLTV, maj_dom)
    mrows.append({"Churn model": m, "AUC": roc_auc_score(y, PROB[m]), "Max profit (S2)": c.max(),
                  "Optimal % targeted": 100 * c.argmax() / N})
mtab = pd.DataFrame(mrows).set_index("Churn model")
mtab.round(3).to_csv(os.path.join(TAB, "T17_profit_by_model.csv"))
print("\n=== Table 17: Does the AUC ranking carry over to profit? ==="); display(mtab.round(3))

# ---------------------------------------------------------------------
# 4. Sensitivity: when does reason-matching start to pay?
# ---------------------------------------------------------------------
G_GRID = [0.20, 0.30, 0.40, 0.50, 0.60]
C_GRID = [50, 100, 200, 400, 800]
sens = {}
for name, od in [("SHAP-guided", shap_dom), ("Why-model", why_dom), ("Oracle", true_dom)]:
    mat = pd.DataFrame(index=[f"γ_match={g}" for g in G_GRID], columns=[f"cost={c}" for c in C_GRID], dtype=float)
    for g in G_GRID:
        for c in C_GRID:
            base = profit_curve(PCAL[PRIMARY] * CLTV, maj_dom, g, G_MISS, c).max()
            alt  = profit_curve(PCAL[PRIMARY] * CLTV, od, g, G_MISS, c).max()
            mat.loc[f"γ_match={g}", f"cost={c}"] = 100 * (alt - base) / abs(base) if base != 0 else np.nan
    sens[name] = mat
    mat.round(1).to_csv(os.path.join(TAB, f"T18_sensitivity_{name}.csv"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
vmax = max(np.nanmax(np.abs(m.values)) for m in sens.values())
for ax, (name, mat) in zip(axes, sens.items()):
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="RdBu", center=0, vmin=-vmax, vmax=vmax, cbar=False, ax=ax)
    ax.set_title(f"{name} offer: % profit gain vs majority offer")
savefig("F19_sensitivity")

# ---------------------------------------------------------------------
# 5. Profit curves
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))
style = {"S1": ("#9AA5B1", "--"), "S2": ("#4A6FA5", "-"), "S2r": ("#9AA5B1", ":"), "S3": ("#C8553D", "-"),
         "S4": ("#5B8C5A", "-"), "S5": ("#2F2F2F", ":"), "S2u": ("#4A6FA5", ":")}
x = np.arange(N + 1) / N * 100
for s, c in curves.items():
    col, ls = style[s.split()[0]]
    ax.plot(x, c / 1000, color=col, ls=ls, lw=1.6, label=s)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("% of customers targeted (ranked)"); ax.set_ylabel("Cumulative profit (thousands)")
ax.set_title("Retention profit curves (out-of-fold)"); ax.legend(frameon=False, fontsize=8)
ax.set_xlim(0, 60)
savefig("F20_profit_curves")

# ---------------------------------------------------------------------
# 6. Retention playbook: who to target and with which offer (why-model offer)
# ---------------------------------------------------------------------
k_opt = curves["S4 Risk×CLTV + why-model offer"].argmax()
tgt = np.argsort(-(PCAL[PRIMARY] * CLTV), kind="stable")[:k_opt]
seg = pd.DataFrame({"Offer (why-model)": [OFFER[d] for d in why_dom[tgt]],
                    "CLTV tier": pd.qcut(CLTV, 3, labels=["Low", "Mid", "High"])[tgt],
                    "Churn prob": PCAL[PRIMARY][tgt], "Churned": y[tgt]})
play = seg.groupby(["Offer (why-model)", "CLTV tier"]).agg(Customers=("Churned", "size"),
                                                           Mean_churn_prob=("Churn prob", "mean"),
                                                           Actual_churn_rate=("Churned", "mean")).round(3)
play.to_csv(os.path.join(TAB, "T19_retention_playbook.csv"))
print(f"\n=== Table 19: Retention playbook — {k_opt} customers targeted ({100*k_opt/N:.1f}%) ==="); display(play)

print("\nSaved to", OUT, "->", sorted(os.listdir(TAB)))
print("\nHOW TO READ:")
print(" Table 16: S3 (SHAP-guided) vs S2 (majority offer) -> does SHAP-based prescription add profit?")
print("           S4 (why-model) and S5 (oracle bound) -> how much a real reason signal is worth")
print(" Table 17: does a higher AUC also give higher profit?")
print(" F19: under which offer-effectiveness / cost settings reason-matching pays off")
