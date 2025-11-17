import re
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from pyod.models.iforest import IForest

# PATHS / SETTINGS
BASE = Path("fl_gaia_zoo_congo_15aug25_data")
MAP_PATH = Path("species_mapping_refined.csv")
META_PATH = BASE / "fl_gaia_zoo_congo_15aug25_data_metadata_with_perch.xlsx"
AVIARY_PATH = Path("aviary_details.xlsx")   # actual aviary details file
OUT_DIR = Path("species_anomaly_results_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 22050
N_MFCC = 20
CONTAMINATION = 0.05
RANDOM_STATE = 0
MIN_SAMPLES = 5
TOP_SPIKES_TO_SAVE = 200


# HELPER FUNCTIONS
def normalize_name(name: str) -> str:
    """Lowercase, remove extra spaces and parentheses – for consistent matching."""
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def robust_z(x: pd.Series) -> pd.Series:
    """Robust Z-score using MAD."""
    x = pd.to_numeric(x, errors="coerce")
    med = x.median()
    mad = (np.abs(x - med)).median()
    if mad == 0 or np.isnan(mad):
        std = x.std()
        return (x - x.mean()) / (std if std else 1)
    return 0.6745 * (x - med) / mad


def extract_features(path: str):
    """Extract MFCC and key spectral features."""
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        if len(y) < 1000:
            return None
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rm = librosa.feature.rms(y=y)
        zc = librosa.feature.zero_crossing_rate(y)
        vec = np.hstack([
            mf.mean(axis=1), mf.std(axis=1),
            sc.mean(), sc.std(),
            sb.mean(), sb.std(),
            ro.mean(), ro.std(),
            rm.mean(), rm.std(),
            zc.mean(), zc.std()
        ])
        return vec
    except Exception:
        return None


def to_basename(x):
    try:
        return Path(str(x)).name
    except Exception:
        return None


# LOAD SPECIES MAPPING + METADATA
print("📂 Loading species mapping and metadata...")
mapping = pd.read_csv(MAP_PATH)
meta = pd.read_excel(META_PATH)

mapping.columns = [c.lower().strip() for c in mapping.columns]
meta.columns = [c.lower().strip() for c in meta.columns]

# create filename keys for merging
mapping["basename"] = (
    mapping["filename"].apply(to_basename)
    if "filename" in mapping.columns
    else mapping.iloc[:, 0].apply(to_basename)
)
meta["basename"] = (
    meta["filename"].apply(to_basename)
    if "filename" in meta.columns
    else meta.iloc[:, 0].apply(to_basename)
)

meta["datetime"] = pd.to_datetime(meta["datetime"], errors="coerce")

# merge detections with metadata
merged = pd.merge(mapping, meta, on="basename", how="inner")

# match basenames to actual local .wav files
audio_files = list(BASE.glob("**/*.wav"))
audio_map = {Path(p).name: str(p) for p in audio_files}
merged["file"] = merged["basename"].map(audio_map)

# LOAD AVIARY DETAILS AND FILTER TO CONGO AVIARY SPECIES
aviary = pd.read_excel(AVIARY_PATH)
aviary.columns = [c.lower().strip() for c in aviary.columns]

# find the right subset
if "aviary" in aviary.columns:
    congo_rows = aviary[aviary["aviary"].str.contains("congo", case=False, na=False)]
else:
    print("⚠️ No 'aviary' column found — assuming all rows are for Congo Aviary")
    congo_rows = aviary.copy()

# detect the species column
possible_cols = ["scientific name", "scientific_name", "species", "species name"]
found_col = None
for col in possible_cols:
    if col in aviary.columns:
        found_col = col
        break

if found_col:
    aviary_species = (
        congo_rows[found_col]
        .dropna()
        .map(normalize_name)
        .unique()
        .tolist()
    )
    print(f"✅ Loaded {len(aviary_species)} Congo Aviary species: {aviary_species}")
else:
    print("❌ Could not find species column in aviary_details.xlsx")
    aviary_species = []

# normalize merged species and filter
merged["mapped_species_norm"] = merged["mapped_species"].map(normalize_name)

if aviary_species:
    merged = merged[merged["mapped_species_norm"].isin(aviary_species)].copy()
    print(f"✅ After filtering to Congo Aviary species: {len(merged)} clips remain")
else:
    print("⚠️ Using all mapped species (aviary species not detected)")

# BASIC CHECKS
merged = merged.dropna(subset=["file", "datetime", "mapped_species"])
print(f"✅ Total valid clips for analysis: {len(merged)}")

if len(merged) == 0:
    raise SystemExit("❌ No valid clips after filtering — check aviary_details.xlsx column names!")

# summary per species
species_counts = merged["mapped_species"].value_counts()
print("\n📊 Clip count per species:")
print(species_counts)

# RUN ANOMALY DETECTION PER SPECIES
results = []
species_list = sorted(merged["mapped_species"].dropna().unique())
print(f"\nDetected {len(species_list)} species for Congo Aviary: {species_list}")

for species in species_list:
    sub = merged[merged["mapped_species"] == species]
    if len(sub) < MIN_SAMPLES:
        print(f"  ↪ Skipping {species} (only {len(sub)} clips)")
        continue

    print(f"\n🔍 Analyzing species: {species} ({len(sub)} clips)")
    X, files, times = [], [], []

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"Featurizing {species[:25]}", leave=False):
        feats = extract_features(row["file"])
        if feats is not None:
            X.append(feats)
            files.append(row["file"])
            times.append(row["datetime"])

    if not X:
        print(f"  ↪ No usable audio for {species}")
        continue

    X = np.stack(X)
    clf = IForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    clf.fit(X)
    scores = clf.decision_function(X)

    df = pd.DataFrame({
        "file": files,
        "datetime": pd.to_datetime(times),
        "anomaly_score": scores,
        "species": species
    })
    results.append(df)

if not results:
    raise SystemExit("❌ No species had enough clips to run anomaly detection.")

anoms = pd.concat(results, ignore_index=True)
anoms.to_csv(OUT_DIR / "species_anomaly_scores.csv", index=False)
print(f"\n✅ Saved anomaly scores to {OUT_DIR / 'species_anomaly_scores.csv'}")

# DETECT STRESS SPIKES (HOURLY)
print("\n⚡ Detecting stress spikes...")
hourly = (
    anoms.set_index("datetime")
         .groupby("species")["anomaly_score"]
         .resample("H").mean()
         .reset_index()
)

spike_rows = []
for sp, grp in hourly.groupby("species"):
    g = grp.copy().sort_values("datetime")
    if g["anomaly_score"].notna().sum() < 8:
        continue
    g["z"] = robust_z(g["anomaly_score"])
    p90 = g["anomaly_score"].quantile(0.9)
    spikes = g[(g["z"] > 2.5) & (g["anomaly_score"] > p90)]
    if not spikes.empty:
        spikes["species"] = sp
        spike_rows.append(spikes[["species", "datetime", "anomaly_score", "z"]])

if spike_rows:
    spikes_all = (
        pd.concat(spike_rows, ignore_index=True)
          .sort_values(["z", "anomaly_score"], ascending=False)
    )
    spikes_top = spikes_all.head(TOP_SPIKES_TO_SAVE)
    spikes_top.to_csv(OUT_DIR / "stress_spikes.csv", index=False)
    print(f"✅ Saved stress spikes to {OUT_DIR / 'stress_spikes.csv'}")
else:
    print("⚠️ No stress spikes detected with current thresholds.")

print("\n🎯 Congo Aviary anomaly detection complete.")
