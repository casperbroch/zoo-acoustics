import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_anomaly_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the anomaly scores CSV file."""
    df = pd.read_csv(csv_path)

    # Ensure correct columns
    expected_cols = {"file", "datetime", "anomaly_score", "species"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Convert datetime to pandas datetime and drop duplicates
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.drop_duplicates()

    # Sort by datetime for correct plotting
    df = df.sort_values("datetime").reset_index(drop=True)

    # Extract recording group from path (e.g., "100" from the file path)
    df["recording_group"] = df["file"].apply(lambda x: Path(x).parts[-2] if isinstance(x, str) else None)

    return df


def plot_anomaly_time_series(df: pd.DataFrame, output_dir: str):
    """Plot anomaly score time series per species."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="datetime", y="anomaly_score", hue="species", marker="o", linewidth=1.2)
    plt.title("Anomaly Scores Over Time per Species", fontsize=14)
    plt.xlabel("Datetime")
    plt.ylabel("Anomaly Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "anomaly_time_series.png", dpi=300)
    plt.close()


def plot_anomaly_distribution(df: pd.DataFrame, output_dir: str):
    """Plot distribution of anomaly scores."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df["anomaly_score"], bins=40, kde=True, color="teal")
    plt.title("Distribution of Anomaly Scores", fontsize=14)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "anomaly_distribution.png", dpi=300)
    plt.close()


def plot_daily_anomaly_trend(df: pd.DataFrame, output_dir: str):
    """Plot average daily anomaly score trend."""
    df["date"] = df["datetime"].dt.date
    daily_avg = df.groupby(["date", "species"])["anomaly_score"].mean().reset_index()

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=daily_avg, x="date", y="anomaly_score", hue="species", marker="o")
    plt.title("Daily Average Anomaly Score Trend", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Average Anomaly Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "daily_anomaly_trend.png", dpi=300)
    plt.close()


def plot_anomaly_scatter(df: pd.DataFrame, output_dir: str):
    """Scatter plot of anomaly scores with rolling mean per species."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="datetime", y="anomaly_score", hue="species", s=20, alpha=0.6)
    for species, sub in df.groupby("species"):
        sub_sorted = sub.sort_values("datetime")
        sub_sorted["rolling_mean"] = sub_sorted["anomaly_score"].rolling(window=10, min_periods=1).mean()
        plt.plot(sub_sorted["datetime"], sub_sorted["rolling_mean"], label=f"{species} (Rolling Avg)", linewidth=2)

    plt.title("Anomaly Scatter with Rolling Mean", fontsize=14)
    plt.xlabel("Datetime")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "anomaly_scatter_rolling.png", dpi=300)
    plt.close()

def plot_weekly_heatmaps(df: pd.DataFrame, output_dir: str):
    """Generate Day × Hour anomaly heatmaps for each species."""
    print("🗺️  Generating weekly anomaly heatmaps...")

    # Extract time components
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour

    # Ensure consistent ordering
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    print(df["species"].value_counts())

    # Loop through species
    for species, sub in df.groupby("species"):
        if len(sub) < 5:
            print(f"⚠️ Skipping {species} — only {len(sub)} data points.")
            continue

        # Pivot to create day × hour matrix
        heatmap_data = (
            sub.groupby(["day_of_week", "hour"])["anomaly_score"]
            .mean()
            .unstack(fill_value=np.nan)
            .reindex(days_order)
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            heatmap_data,
            cmap="coolwarm",
            center=0,
            linewidths=0.4,
            cbar_kws={"label": "Mean Anomaly Score"},
        )
        plt.title(f"{species} — Mean Anomaly by Day × Hour", fontsize=14, weight="bold")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        plt.tight_layout()

        fname = f"heatmap_{species.replace(' ', '_')}.png"
        plt.savefig(Path(output_dir) / fname, dpi=300)
        plt.close()

    print("✅ Weekly heatmaps saved.")


def generate_anomaly_visualizations(csv_path: str, output_dir: str = "outputs/anomaly_visuals"):
    """Generate all anomaly visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_anomaly_data(csv_path)

    print(f"Loaded {len(df)} rows for {df['species'].nunique()} species.")

    plot_anomaly_time_series(df, output_path)
    plot_anomaly_distribution(df, output_path)
    plot_daily_anomaly_trend(df, output_path)
    plot_anomaly_scatter(df, output_path)
    plot_weekly_heatmaps(df, output_path)  

    print(f"Visualizations saved to: {output_path.resolve()}")

if __name__ == "__main__":
    csv_path = "species_anomaly_results_final/species_anomaly_scores.csv"
    output_dir = "anomaly_plots_final"
    generate_anomaly_visualizations(csv_path, output_dir)