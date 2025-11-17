import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path
import re
import seaborn as sns
import matplotlib.pyplot as plt

# PATHS
BASE = Path("fl_gaia_zoo_congo_15aug25_data")
PRED_PATH = BASE / "perch_predictions_top1.csv"
AVIARY_PATH = "aviary_details.xlsx"
OUT_PATH = "species_mapping_refined.csv"

# LOAD FILES
print("📂 Loading input files...")
pred = pd.read_csv(PRED_PATH)
aviary = pd.read_excel(AVIARY_PATH)

pred.columns = [c.strip().lower().replace(" ", "_") for c in pred.columns]
aviary.columns = [c.strip().lower().replace(" ", "_") for c in aviary.columns]

if "scientific_name" not in pred.columns:
    raise KeyError("Column 'scientific_name' missing in perch_predictions_top1.csv")
if "scientific_name" not in aviary.columns:
    raise KeyError("Column 'Scientific name' missing in aviary_details.xlsx")

print(f"✅ Loaded {len(pred)} detections and {len(aviary)} known zoo species.")

# NORMALIZE SPECIES NAMES
def normalize(name):
    if pd.isna(name): return ""
    name = str(name).lower()
    name = re.sub(r"[_\-]", " ", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()

pred["norm_name"] = pred["scientific_name"].apply(normalize)
aviary["norm_name"] = aviary["scientific_name"].apply(normalize)

pred["genus"] = pred["norm_name"].apply(lambda s: s.split(" ")[0] if s else "")
aviary["genus"] = aviary["norm_name"].apply(lambda s: s.split(" ")[0] if s else "")

detected_species = sorted(pred["norm_name"].dropna().unique())
known_species = sorted(aviary["norm_name"].dropna().unique())

print(f"Detected unique species: {len(detected_species)}")
print(f"Known zoo species: {len(known_species)}")

# FUZZY MATCHING
mapping = {}
for det in detected_species:
    genus = det.split(" ")[0] if det else ""
    best_match = None
    best_score = 0

    # Genus-prioritized subset first
    genus_subset = aviary[aviary["genus"] == genus]
    if len(genus_subset):
        match, score, _ = process.extractOne(
            det, genus_subset["norm_name"], scorer=fuzz.token_sort_ratio
        )
    else:
        match, score, _ = process.extractOne(
            det, known_species, scorer=fuzz.token_sort_ratio
        )

    # High-confidence direct matches
    if score >= 70:
        best_match = match
    # Moderate fuzzy match, same genus or partial token overlap
    elif score >= 50 and (
        genus in match or any(w in match for w in det.split(" "))
    ):
        best_match = match
    # Loose fallback — if genus appears anywhere in known list
    elif score >= 40 and genus in " ".join(known_species):
        best_match = match
    else:
        best_match = "Other/Unknown"

    mapping[det] = best_match

pred["mapped_species"] = pred["norm_name"].map(mapping).fillna("Other/Unknown")

# VALIDATION RULES
def match_score(row):
    if row["mapped_species"].lower() in ["other/unknown", "other", "unknown", ""]:
        return 0
    return fuzz.token_sort_ratio(row["norm_name"], row["mapped_species"])

pred["match_score"] = pred.apply(match_score, axis=1)
pred["genus_mapped"] = pred["mapped_species"].apply(lambda s: s.split(" ")[0] if s else "")

# Acceptance criteria
def is_valid(row):
    if row["match_score"] >= 60:
        return True
    if row["genus"] == row["genus_mapped"] and row["match_score"] >= 45:
        return True
    return False

pred["is_valid"] = pred.apply(is_valid, axis=1)
pred.loc[~pred["is_valid"], "mapped_species"] = "Other/Unknown"

valid_ratio = pred["is_valid"].mean() * 100
print(f"✅ {valid_ratio:.1f}% of mappings kept as valid (looser threshold).")

# SUMMARY & DIAGNOSTICS
summary = (
    pred["mapped_species"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "species", "mapped_species": "count"})
)
print("\nTop mapped species:")
print(summary.head(20))

# Plot match score distribution
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.histplot(pred["match_score"], bins=30, kde=True, color="seagreen")
plt.title("Fuzzy Match Score Distribution (Relaxed Mapping)")
plt.xlabel("Match score (0–100)")
plt.ylabel("Number of detections")
plt.tight_layout()
plt.savefig("mapping_quality_distribution.png", dpi=160)
plt.close()

# Save uncertain mappings
mismatches = pred[(pred["is_valid"] == False) & (pred["match_score"] > 0)]
mismatches = mismatches.sort_values("match_score", ascending=False)
mismatches[["scientific_name", "mapped_species", "match_score"]].head(40).to_csv(
    "questionable_mappings.csv", index=False
)

# SAVE FINAL OUTPUT
pred.to_csv(OUT_PATH, index=False)
print(f"\n✅ Refined mapping saved to: {OUT_PATH}")
print(f"📊 Match score histogram: {'mapping_quality_distribution.png'}")
print(f"⚠️ Questionable mappings: {'questionable_mappings.csv'}")

print("\n🎯 Flexible species mapping pipeline complete!")
