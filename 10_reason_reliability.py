# =====================================================================
# NB10 (single cell): How reliable are the benchmark's stated reasons?
#   A. Same customers, two releases: do churn labels, features and stated
#      reasons agree between the 33-variable sample (Study 1) and the
#      11.1.3+ release (Study 2)?
#   B. Are reasons clustered by location? Share of churner pairs in the same
#      ZIP code / city with the same reason group vs a permutation null.
# RUNTIME: < 1 min.  Output: /kaggle/working/nb10/
# =====================================================================
import os, glob
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")  # outputs; later steps read earlier outputs here
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")    # raw datasets (Kaggle input or a local folder)
try:
    display
except NameError:  # running as a plain script outside Jupyter
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

SEED, N_PERM = 42, 2000
import os, re, glob, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
rng = np.random.default_rng(SEED)
OUT = os.path.join(WORK_DIR, "nb10"); os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
files = glob.glob(DATA_DIR + "/**/*.xlsx", recursive=True) + glob.glob(DATA_DIR + "/**/*.csv", recursive=True)
def load(pred):
    for p in files:
        d = pd.read_excel(p) if p.endswith(".xlsx") else pd.read_csv(p)
        if pred({norm(c) for c in d.columns}): return p, d
    raise FileNotFoundError
p1, A = load(lambda c: "churnreason" in c and "totalextradatacharges" not in c)
p2, B = load(lambda c: "totalextradatacharges" in c and "churnreason" in c)
print("Release 1:", p1, A.shape, "\nRelease 2:", p2, B.shape)
def col(d, *names):
    m = {norm(c): c for c in d.columns}
    return next(m[norm(n)] for n in names if norm(n) in m)
GROUPS = {"Competitor": ["Competitor offered higher download speeds", "Competitor offered more data", "Competitor made better offer", "Competitor had better devices"],
          "Price": ["Price too high", "Extra data charges", "Long distance charges", "Lack of affordable download/upload speed"],
          "Support": ["Attitude of support person", "Attitude of service provider", "Poor expertise of phone support", "Poor expertise of online support"],
          "Quality": ["Network reliability", "Product dissatisfaction", "Service dissatisfaction", "Lack of self-service on Website", "Limited range of services"],
          "Uncontrollable": ["Moved", "Deceased"], "Unknown": ["Don't know"]}
R2G = {r.lower(): g for g, rs in GROUPS.items() for r in rs}
grp = lambda s: s.astype(str).str.strip().str.replace("’", "'").str.lower().map(R2G)

a = pd.DataFrame({"id": A[col(A, "CustomerID", "Customer ID")], "churn1": A[col(A, "Churn Label")].astype(str).str.lower().eq("yes"),
                  "reason1": A[col(A, "Churn Reason")], "tenure1": A[col(A, "Tenure Months")], "mc1": A[col(A, "Monthly Charges")],
                  "zip1": A[col(A, "Zip Code")], "city1": A[col(A, "City")]})
b = pd.DataFrame({"id": B[col(B, "Customer ID", "CustomerID")], "churn2": B[col(B, "Churn Label")].astype(str).str.lower().eq("yes"),
                  "reason2": B[col(B, "Churn Reason")], "tenure2": B[col(B, "Tenure in Months", "Tenure Months")],
                  "mc2": B[col(B, "Monthly Charge", "Monthly Charges")]})
m = a.merge(b, on="id"); print(f"\nMatched customers: {len(m)}")

# ---------- A. agreement between releases
both = m[m.churn1 & m.churn2].copy()
both["g1"], both["g2"] = grp(both.reason1), grp(both.reason2)
same_reason = (both.reason1.astype(str).str.strip().str.lower() == both.reason2.astype(str).str.strip().str.lower())
# chance agreement if reasons were assigned independently with the observed marginals
p1r = both.reason1.astype(str).str.lower().value_counts(normalize=True); p2r = both.reason2.astype(str).str.lower().value_counts(normalize=True)
p1g = both.g1.value_counts(normalize=True); p2g = both.g2.value_counts(normalize=True)
rowsA = {"Customers matched by ID": len(m),
         "Churn label identical (%)": 100 * (m.churn1 == m.churn2).mean(),
         "Tenure identical (%)": 100 * (m.tenure1 == m.tenure2).mean(),
         "Monthly charge identical (±0.01) (%)": 100 * (np.abs(m.mc1 - m.mc2) <= 0.01).mean(),
         "Churners in both releases": len(both),
         "Same stated reason (%)": 100 * same_reason.mean(),
         "  expected if independent (%)": 100 * (p1r * p2r.reindex(p1r.index).fillna(0)).sum(),
         "Same reason group (%)": 100 * (both.g1 == both.g2).mean(),
         "  expected if independent (%) ": 100 * (p1g * p2g.reindex(p1g.index).fillna(0)).sum(),
         "Cohen's κ, reason group": cohen_kappa_score(both.g1.fillna("NA"), both.g2.fillna("NA"))}
TA = pd.Series(rowsA).round(3).to_frame("Value"); TA.to_csv(f"{OUT}/T30_release_agreement.csv")
print("\n=== A. Same customers, two releases ==="); display(TA)
ct = pd.crosstab(both.g1.rename("Release 1 (Study 1)"), both.g2.rename("Release 2 (Study 2)"))
ct.to_csv(f"{OUT}/T31_reason_group_crosstab.csv"); display(ct)

# ---------- B. geographic clustering of reasons (Study 1 churners)
ch = a[a.churn1].copy(); ch["g"] = grp(ch.reason1)
def same_pair_share(key, labels):
    df = pd.DataFrame({"k": key.values, "g": labels})
    s, n = 0, 0
    for _, g in df.groupby("k"):
        c = g["g"].value_counts().values; t = len(g)
        if t < 2: continue
        s += (c * (c - 1)).sum() / 2; n += t * (t - 1) / 2
    return s / n if n else np.nan
rowsB = []
for name, key in [("ZIP code", ch.zip1), ("City", ch.city1)]:
    obs = same_pair_share(key, ch.g.values)
    null = np.array([same_pair_share(key, rng.permutation(ch.g.values)) for _ in range(N_PERM // 10)])
    rowsB.append({"Grouping": name, "Groups with ≥2 churners": int((key.value_counts() >= 2).sum()),
                  "Same-group pairs, observed (%)": round(100 * obs, 1), "Permutation mean (%)": round(100 * null.mean(), 1),
                  "Permutation p": round((np.sum(null >= obs) + 1) / (len(null) + 1), 4)})
TB = pd.DataFrame(rowsB).set_index("Grouping"); TB.to_csv(f"{OUT}/T32_geo_clustering.csv")
print("\n=== B. Do churners in the same location state the same reason group? ==="); display(TB)
print("Saved to", OUT)
