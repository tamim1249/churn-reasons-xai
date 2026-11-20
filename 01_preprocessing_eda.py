# =====================================================================
# NB1 (single cell): Leakage-Free Preprocessing, Reason Grouping & EDA
# Paper: Predicting Who Leaves Is Not Knowing Why: Validating Churn-Model
#        Explanations Against Customers' Stated Reasons
# Output: /kaggle/working/telco_clean.csv, config.json, figures/, tables/
# =====================================================================


# ---------------------------------------------------------------------
# NB1 — Leakage-Free Preprocessing, Reason Grouping & EDA
# ---------------------------------------------------------------------
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_PATH = os.environ.get("NB_DATA") or next(iter(glob.glob(os.path.join(DATA_DIR, "**", "Telco_customer_churn.xlsx"), recursive=True)), "Telco_customer_churn.xlsx")
OUT_DIR   = os.environ.get("NB_OUT", WORK_DIR)
FIG_DIR   = os.path.join(OUT_DIR, "figures")
TAB_DIR   = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
SEED = 42

# Paper-ready plot style: churn = warm, retained = neutral blue
C_CHURN, C_STAY, C_ACCENT = "#C8553D", "#4A6FA5", "#2F2F2F"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight")
    plt.show()

def savetab(df, name, index=True):
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=index)
    return df

# ---------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------
raw = pd.read_excel(DATA_PATH, sheet_name="Telco_Churn")
print("Raw shape:", raw.shape)
display(raw.head(3))

# ---------------------------------------------------------------------
# 2. Column roles
# ---------------------------------------------------------------------
ID_COL, TARGET, VALUE_COL = "CustomerID", "Churn Value", "CLTV"
CONSTANT_COLS  = [c for c in raw.columns if raw[c].nunique(dropna=False) == 1]
REDUNDANT_COLS = ["Lat Long", "City", "Zip Code"]
LEAKAGE_COLS   = ["Churn Label", "Churn Score", "Churn Reason"]
print("Constant :", CONSTANT_COLS)
print("Redundant:", REDUNDANT_COLS)
print("Leakage  :", LEAKAGE_COLS)

# ---------------------------------------------------------------------
# 3. Leakage audit (paper er jonno Table)
# ---------------------------------------------------------------------
y_raw = raw[TARGET].astype(int)
audit_rows = [
    ("Churn Label",               roc_auc_score(y_raw, (raw["Churn Label"] == "Yes").astype(int)), "Target er copy"),
    ("Churn Score",               roc_auc_score(y_raw, raw["Churn Score"]),                        "IBM SPSS model er predicted score"),
    ("Churn Reason (non-null)",   roc_auc_score(y_raw, raw["Churn Reason"].notna().astype(int)),   "Shudhu churner der e reason ache"),
    ("CLTV",                      roc_auc_score(y_raw, raw["CLTV"]),                               "Business value — feature hishebe deya hobe na"),
]
leak = pd.DataFrame(audit_rows, columns=["Column", "Standalone AUC", "Interpretation"]).set_index("Column")
leak["Standalone AUC"] = leak["Standalone AUC"].round(4)
print(pd.crosstab(raw["Churn Reason"].notna(), raw["Churn Label"], rownames=["Reason present"]))
display(savetab(leak, "T_leakage_audit"))

# ---------------------------------------------------------------------
# 4. Cleaning
# ---------------------------------------------------------------------
df = raw.copy()

# 4.1 Total Charges: text -> numeric
tc = pd.to_numeric(df["Total Charges"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")
blank = tc.isna()
print("Blank Total Charges:", int(blank.sum()))
print(df.loc[blank, ["Tenure Months", "Monthly Charges", TARGET]].value_counts().head(15))
print("Customers with Tenure = 0 (overall):", int((df["Tenure Months"] == 0).sum()))

if (df.loc[blank, "Tenure Months"] == 0).all():
    tc[blank] = 0.0                       # new customers, not billed yet
    print("-> All blanks have tenure 0: filled with 0 (not yet billed).")
else:
    tc[blank] = df.loc[blank, "Monthly Charges"] * df.loc[blank, "Tenure Months"]
    print("-> Some blanks have tenure > 0: filled with Monthly x Tenure. CHECK MANUALLY.")
df["Total Charges"] = tc.astype(float)

# 4.2 Yes/No -> 1/0 ; Gender -> Male (1/0)
BIN_FEATURES = [c for c in df.columns
                if c not in LEAKAGE_COLS + [TARGET]
                and set(df[c].dropna().unique()) <= {"Yes", "No"}]
for c in BIN_FEATURES:
    df[c] = (df[c] == "Yes").astype(int)
df["Male"] = (df["Gender"] == "Male").astype(int)
BIN_FEATURES = BIN_FEATURES + ["Male"]

# 4.3 Multi-category features stay as strings (NB2 pipeline e encode hobe, fold er bhitore)
CAT_FEATURES = ["Multiple Lines", "Internet Service", "Online Security", "Online Backup",
                "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
                "Contract", "Payment Method"]
for c in CAT_FEATURES:
    df[c] = df[c].astype(str).str.strip()

print("Binary:", BIN_FEATURES)
print("Categorical:", CAT_FEATURES)

# ---------------------------------------------------------------------
# 4.4 Light feature engineering (row-wise, tai leakage nai)
# ---------------------------------------------------------------------
SERVICE_COLS = ["Online Security", "Online Backup", "Device Protection", "Tech Support",
                "Streaming TV", "Streaming Movies"]
df["Num Services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + df["Phone Service"] + (df["Multiple Lines"] == "Yes").astype(int)
df["Avg Monthly Spend"] = np.where(df["Tenure Months"] > 0,
                                   df["Total Charges"] / df["Tenure Months"].replace(0, np.nan),
                                   df["Monthly Charges"])
df["Charge Gap"] = df["Monthly Charges"] - df["Avg Monthly Spend"]

NUM_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges", "Latitude", "Longitude",
                "Num Services", "Avg Monthly Spend", "Charge Gap"]
display(df[NUM_FEATURES].describe().T.round(2))

# ---------------------------------------------------------------------
# 5. Churn Reason → Reason Group (validation ground truth)
# ---------------------------------------------------------------------
REASON_GROUPS = {
    "Competitor": ["Competitor offered higher download speeds", "Competitor offered more data",
                   "Competitor made better offer", "Competitor had better devices"],
    "Price":      ["Price too high", "Extra data charges", "Long distance charges",
                   "Lack of affordable download/upload speed"],
    "Support":    ["Attitude of support person", "Attitude of service provider",
                   "Poor expertise of phone support", "Poor expertise of online support"],
    "Quality":    ["Network reliability", "Product dissatisfaction", "Service dissatisfaction",
                   "Lack of self-service on Website", "Limited range of services"],
    "Uncontrollable": ["Moved", "Deceased"],
    "Unknown":    ["Don't know"],
}
reason_to_group = {r: g for g, rs in REASON_GROUPS.items() for r in rs}

df["Churn Reason"] = df["Churn Reason"].astype("string").str.strip().str.replace("\u2019", "'", regex=False)
df["Reason Group"] = df["Churn Reason"].map(reason_to_group)

unmapped = df.loc[df["Churn Reason"].notna() & df["Reason Group"].isna(), "Churn Reason"].unique()
assert len(unmapped) == 0, f"Unmapped reasons: {unmapped}"
assert df.loc[df[TARGET] == 1, "Reason Group"].notna().all(), "Some churners have no reason"

df["Controllable"] = np.where(df[TARGET] == 0, np.nan,
                              (~df["Reason Group"].isin(["Uncontrollable"])).astype(float))

rg = (df.loc[df[TARGET] == 1, "Reason Group"].value_counts()
        .rename("Count").to_frame())
rg["% of churners"] = (rg["Count"] / rg["Count"].sum() * 100).round(1)
rg["CLTV lost (sum)"] = df.loc[df[TARGET] == 1].groupby("Reason Group")[VALUE_COL].sum()
rg["Mean CLTV"] = df.loc[df[TARGET] == 1].groupby("Reason Group")[VALUE_COL].mean().round(0)
display(savetab(rg, "T_reason_groups"))

# ---------------------------------------------------------------------
# 6. Final feature set + save
# ---------------------------------------------------------------------
FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES
assert not set(FEATURES) & set(LEAKAGE_COLS + [VALUE_COL, TARGET]), "Leakage column in features!"

KEEP = [ID_COL] + FEATURES + [TARGET, VALUE_COL, "Churn Reason", "Reason Group", "Controllable"]
clean = df[KEEP].copy()
clean.to_csv(os.path.join(OUT_DIR, "telco_clean.csv"), index=False)

config = {
    "id_col": ID_COL, "target": TARGET, "value_col": VALUE_COL,
    "num_features": NUM_FEATURES, "bin_features": BIN_FEATURES, "cat_features": CAT_FEATURES,
    "features": FEATURES, "leakage_cols": LEAKAGE_COLS,
    "dropped_cols": CONSTANT_COLS + REDUNDANT_COLS + ["Gender"],
    "reason_groups": REASON_GROUPS, "seed": SEED,
}
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("Saved telco_clean.csv:", clean.shape)
print("n features:", len(FEATURES), "| churn rate:", round(clean[TARGET].mean(), 4))

# ---------------------------------------------------------------------
# 7. Table 1 — Descriptive statistics by churn status
# ---------------------------------------------------------------------
rows = []
for c in NUM_FEATURES + [VALUE_COL]:
    s0, s1 = df.loc[df[TARGET] == 0, c], df.loc[df[TARGET] == 1, c]
    rows.append((c, f"{s0.mean():.2f} ± {s0.std():.2f}", f"{s1.mean():.2f} ± {s1.std():.2f}"))
for c in BIN_FEATURES:
    rows.append((c + " (%)", f"{df.loc[df[TARGET]==0, c].mean()*100:.1f}", f"{df.loc[df[TARGET]==1, c].mean()*100:.1f}"))
t1 = pd.DataFrame(rows, columns=["Variable", f"Retained (n={int((df[TARGET]==0).sum())})",
                                 f"Churned (n={int((df[TARGET]==1).sum())})"]).set_index("Variable")
display(savetab(t1, "T1_descriptives"))

# ---------------------------------------------------------------------
# 8. EDA figures
# ---------------------------------------------------------------------
# Fig 1: churn rate by key categorical drivers
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
overall = df[TARGET].mean() * 100
for ax, col in zip(axes.ravel(), ["Contract", "Internet Service", "Payment Method", "Tech Support"]):
    rate = df.groupby(col)[TARGET].mean().mul(100).sort_values()
    ax.barh(rate.index, rate.values, color=C_CHURN)
    ax.axvline(overall, ls="--", lw=1, color=C_ACCENT)
    for i, v in enumerate(rate.values):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_title(col); ax.set_xlabel("Churn rate (%)"); ax.set_xlim(0, rate.max() * 1.2)
fig.suptitle(f"Churn rate by customer attribute (dashed = overall {overall:.1f}%)", y=1.01)
savefig("F1_churn_rate_by_category")
# Fig 2: tenure, monthly charges, charge gap distributions
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col in zip(axes, ["Tenure Months", "Monthly Charges", "Charge Gap"]):
    for val, colr, lab in [(0, C_STAY, "Retained"), (1, C_CHURN, "Churned")]:
        sns.kdeplot(df.loc[df[TARGET] == val, col], ax=ax, color=colr, fill=True, alpha=0.25, lw=1.5, label=lab, common_norm=False)
    ax.set_title(col); ax.set_ylabel("Density")
axes[0].legend(frameon=False)
savefig("F2_numeric_distributions")
# Fig 3: reason groups (count) + 20 detailed reasons
churners = df[df[TARGET] == 1]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.6]})
g = churners["Reason Group"].value_counts().sort_values()
axes[0].barh(g.index, g.values, color=C_CHURN)
for i, v in enumerate(g.values):
    axes[0].text(v + 5, i, f"{v} ({v/len(churners)*100:.0f}%)", va="center", fontsize=9)
axes[0].set_title("Churn reason groups"); axes[0].set_xlim(0, g.max() * 1.3)

r = churners["Churn Reason"].value_counts().sort_values()
axes[1].barh(r.index, r.values, color=[C_STAY] * len(r))
axes[1].set_title("Stated churn reasons (n = %d churners)" % len(churners))
savefig("F3_churn_reasons")
# Fig 4: reason group x contract / internet service (row %): which customers leave for which reason
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, col in zip(axes, ["Contract", "Internet Service"]):
    ct = pd.crosstab(churners["Reason Group"], churners[col], normalize="index").mul(100)
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Reason group × {col} (row %)"); ax.set_ylabel("")
savefig("F4_reason_by_profile")
# Fig 5: geographic churn map (zip-level, >= 5 customers)
geo = (raw.assign(churn=raw[TARGET])
          .groupby("Zip Code").agg(lat=("Latitude", "mean"), lon=("Longitude", "mean"),
                                   n=("churn", "size"), rate=("churn", "mean")))
geo = geo[geo["n"] >= 5]
fig, ax = plt.subplots(figsize=(6.5, 7.5))
sc = ax.scatter(geo["lon"], geo["lat"], c=geo["rate"] * 100, s=geo["n"] * 2,
                cmap="RdYlBu_r", alpha=0.8, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Churn rate (%)", shrink=0.7)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"Zip-level churn rate, California ({len(geo)} zips with ≥5 customers)")
savefig("F5_geo_churn_map")
# Fig 6: CLTV at risk by reason group
fig, ax = plt.subplots(figsize=(8, 4))
order = churners.groupby("Reason Group")[VALUE_COL].median().sort_values().index
sns.boxplot(data=churners, y="Reason Group", x=VALUE_COL, order=order, color=C_STAY, fliersize=2, ax=ax)
ax.axvline(df.loc[df[TARGET] == 0, VALUE_COL].median(), ls="--", color=C_ACCENT, lw=1)
ax.set_title("CLTV of churned customers by reason group (dashed = retained median)"); ax.set_ylabel("")
savefig("F6_cltv_by_reason")

# ---------------------------------------------------------------------
# 9. Draft: feature → reason-domain map (NB3 er jonno)
# ---------------------------------------------------------------------
FEATURE_DOMAIN = {
    "Price":   ["Monthly Charges", "Total Charges", "Avg Monthly Spend", "Charge Gap"],
    "Support": ["Tech Support", "Online Security", "Device Protection"],
    "Quality/Competitor (offering)": ["Internet Service", "Streaming TV", "Streaming Movies",
                                      "Multiple Lines", "Online Backup", "Phone Service", "Num Services"],
    "Relationship (no direct reason)": ["Contract", "Tenure Months", "Paperless Billing", "Payment Method"],
    "Demographic/Geo (no direct reason)": ["Male", "Senior Citizen", "Partner", "Dependents", "Latitude", "Longitude"],
}
mapped = {f for fs in FEATURE_DOMAIN.values() for f in fs}
print("Features not mapped:", set(FEATURES) - mapped)
config["feature_domain_draft"] = FEATURE_DOMAIN
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("Outputs:", sorted(os.listdir(OUT_DIR)))
print("Figures:", sorted(os.listdir(FIG_DIR)))
print("Tables :", sorted(os.listdir(TAB_DIR)))
