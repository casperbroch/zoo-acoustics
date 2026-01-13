"""Core data processing functions extracted from notebook analysis"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from api.config import (
    DATA_DIR, NIGHTTIME_HOURS, TIME_BIN_MINUTES,
    ZOOS_CONFIG, SPECIES_INFO
)


def load_zoo_enclosure_data(zoo_name: str, enclosure_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load full and target species data for a specific zoo/enclosure.
    
    Args:
        zoo_name: Name of the zoo
        enclosure_name: Name of the enclosure
        
    Returns:
        Tuple of (df_full_clean, df_target) DataFrames
        
    Raises:
        FileNotFoundError: If data files don't exist
        KeyError: If zoo/enclosure not configured
    """
    if zoo_name not in ZOOS_CONFIG:
        raise KeyError(f"Zoo '{zoo_name}' not found in configuration")
    
    if enclosure_name not in ZOOS_CONFIG[zoo_name]['enclosures']:
        raise KeyError(f"Enclosure '{enclosure_name}' not found for zoo '{zoo_name}'")
    
    config = ZOOS_CONFIG[zoo_name]['enclosures'][enclosure_name]
    data_path = DATA_DIR / config['data_dir']
    
    full_data_path = data_path / config['full_data_file']
    target_data_path = data_path / config['target_data_file']
    
    if not full_data_path.exists():
        raise FileNotFoundError(f"Full data file not found: {full_data_path}")
    if not target_data_path.exists():
        raise FileNotFoundError(f"Target data file not found: {target_data_path}")
    
    # Load datasets
    df_full = pd.read_excel(full_data_path)
    df_target = pd.read_excel(target_data_path)
    
    # Convert datetime columns
    df_full['datetime'] = pd.to_datetime(df_full['datetime'])
    df_target['datetime'] = pd.to_datetime(df_target['datetime'])
    
    # Clean full dataset (remove invalid labels)
    df_full_clean = df_full[
        ~df_full['MIT_AST_label'].isin(['speech_removed', 'file_missing'])
    ].copy()
    
    # Add temporal features to target dataset
    df_target['hour'] = df_target['datetime'].dt.hour
    df_target['date'] = df_target['datetime'].dt.date
    
    return df_full_clean, df_target


def get_available_dates(df_target: pd.DataFrame) -> List[Dict]:
    """
    Get all available dates with nighttime bird detections.
    
    Args:
        df_target: Target species detections DataFrame
        
    Returns:
        List of dicts with date and detection count
    """
    # Filter for nighttime hours
    df_night = df_target[df_target['hour'].isin(NIGHTTIME_HOURS)].copy()
    
    # Calculate night_date (clips after midnight belong to previous night)
    df_night['night_date'] = df_night['datetime'].apply(
        lambda x: x.date() if x.hour >= 20 else (x - timedelta(days=1)).date()
    )
    
    # Group by night_date and count
    date_counts = df_night.groupby('night_date').size().reset_index(name='count')
    
    return [
        {'date': row['night_date'], 'detection_count': row['count']}
        for _, row in date_counts.iterrows()
    ]


def validate_date_exists(target_date: date, df_target: pd.DataFrame) -> bool:
    """
    Check if a specific date has nighttime activity data.
    
    Args:
        target_date: The date to check
        df_target: Target species detections DataFrame
        
    Returns:
        True if date has data, False otherwise
    """
    available_dates = get_available_dates(df_target)
    return target_date in [d['date'] for d in available_dates]


def filter_nighttime_data(
    target_date: date,
    df_full_clean: pd.DataFrame,
    df_target: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, datetime, datetime]:
    """
    Filter datasets for a specific night's window (8 PM to 3 AM).
    
    Args:
        target_date: The night date (starts 8 PM this date)
        df_full_clean: Full cleaned dataset
        df_target: Target species dataset
        
    Returns:
        Tuple of (night_full_data, night_bird_data, night_start, night_end)
    """
    # Define nighttime window: 8 PM on target_date to 3 AM next day
    night_start = pd.Timestamp(f"{target_date} 20:00:00")
    night_end = pd.Timestamp(f"{target_date}") + timedelta(days=1, hours=3)
    
    # Filter full dataset for this time window
    night_full_data = df_full_clean[
        (df_full_clean['datetime'] >= night_start) &
        (df_full_clean['datetime'] <= night_end)
    ].copy()
    
    # Filter target dataset for nighttime hours
    df_target_night = df_target[df_target['hour'].isin(NIGHTTIME_HOURS)].copy()
    
    # Calculate night_date for filtering
    df_target_night['night_date'] = df_target_night['datetime'].apply(
        lambda x: x.date() if x.hour >= 20 else (x - timedelta(days=1)).date()
    )
    
    # Filter for specific night
    night_bird_data = df_target_night[
        df_target_night['night_date'] == target_date
    ].copy()
    night_bird_data = night_bird_data.sort_values('datetime')
    
    return night_full_data, night_bird_data, night_start, night_end


def create_time_bins(night_start: datetime, night_end: datetime) -> pd.DatetimeIndex:
    """
    Create time bins for the nighttime window.
    
    Args:
        night_start: Start of night window
        night_end: End of night window
        
    Returns:
        DatetimeIndex with bin edges
    """
    return pd.date_range(
        start=night_start,
        end=night_end,
        freq=f'{TIME_BIN_MINUTES}min'
    )


def compute_species_activity(
    night_bird_data: pd.DataFrame,
    time_bins: pd.DatetimeIndex,
    species_info: Dict
) -> Dict[str, np.ndarray]:
    """
    Compute bird detections per species per time bin.
    
    Args:
        night_bird_data: Filtered nighttime bird detections
        time_bins: Time bin edges
        species_info: Species information dictionary
        
    Returns:
        Dict mapping species name to counts array
    """
    bird_activity = {}
    
    for species in night_bird_data['detected_species'].unique():
        species_data = night_bird_data[
            night_bird_data['detected_species'] == species
        ]
        counts, _ = np.histogram(species_data['datetime'], bins=time_bins)
        bird_activity[species] = counts
    
    return bird_activity


def compute_overall_activity(
    night_full_data: pd.DataFrame,
    time_bins: pd.DatetimeIndex
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute overall acoustic activity per time bin.
    
    Args:
        night_full_data: Filtered nighttime full dataset
        time_bins: Time bin edges
        
    Returns:
        Tuple of (counts, normalized_intensities)
    """
    counts, _ = np.histogram(night_full_data['datetime'], bins=time_bins)
    
    # Normalize intensities (0-1 scale)
    max_count = max(counts) if max(counts) > 0 else 1
    intensities = counts / max_count
    
    return counts, intensities


def compute_mit_breakdown(
    night_full_data: pd.DataFrame,
    time_bins: pd.DatetimeIndex,
    top_n: int = 8
) -> Tuple[List[str], Dict[int, Dict[str, int]]]:
    """
    Compute MIT sound classification breakdown per time bin.
    
    Args:
        night_full_data: Filtered nighttime full dataset
        time_bins: Time bin edges
        top_n: Number of top MIT labels to include
        
    Returns:
        Tuple of (top_mit_labels, mit_categories_per_bin)
    """
    # Get top N most common MIT classifications for this night
    top_mit_labels = night_full_data['MIT_AST_label'].value_counts().head(top_n).index.tolist()
    
    # Count MIT classifications per bin
    mit_categories = {}
    for i in range(len(time_bins) - 1):
        bin_start = time_bins[i]
        bin_end = time_bins[i + 1]
        
        bin_data = night_full_data[
            (night_full_data['datetime'] >= bin_start) &
            (night_full_data['datetime'] < bin_end)
        ]
        
        mit_categories[i] = bin_data['MIT_AST_label'].value_counts().to_dict()
    
    return top_mit_labels, mit_categories


def find_midnight_index(time_bins: pd.DatetimeIndex) -> Optional[int]:
    """
    Find the index of the time bin containing midnight.
    
    Args:
        time_bins: Time bin edges
        
    Returns:
        Index of midnight bin, or None if not found
    """
    bin_centers = time_bins[:-1] + (time_bins[1] - time_bins[0]) / 2
    
    for i, bc in enumerate(bin_centers):
        if bc.hour == 0 and bc.minute == 0:
            return i
    
    return None


def compute_summary_statistics(
    night_bird_data: pd.DataFrame,
    night_full_data: pd.DataFrame,
    species_info: Dict
) -> Dict:
    """
    Compute summary statistics for the night.
    
    Args:
        night_bird_data: Filtered nighttime bird detections
        night_full_data: Filtered nighttime full dataset
        species_info: Species information dictionary
        
    Returns:
        Dict with summary statistics
    """
    summary = {
        'total_bird_detections': len(night_bird_data),
        'total_sound_clips': len(night_full_data),
        'species_breakdown': {},
        'top_5_mit_labels': []
    }
    
    # Peak bird activity hour
    if len(night_bird_data) > 0:
        mode_hours = night_bird_data['datetime'].dt.hour.mode()
        summary['peak_bird_activity_hour'] = int(mode_hours.values[0]) if len(mode_hours) > 0 else None
    else:
        summary['peak_bird_activity_hour'] = None
    
    # Species breakdown
    for species in night_bird_data['detected_species'].unique():
        species_count = len(night_bird_data[night_bird_data['detected_species'] == species])
        species_pct = (species_count / len(night_bird_data)) * 100 if len(night_bird_data) > 0 else 0
        
        summary['species_breakdown'][species] = {
            'common_name': species_info.get(species, {}).get('common', species),
            'count': species_count,
            'percentage': round(species_pct, 1)
        }
    
    # Top 5 MIT sound types
    if len(night_full_data) > 0:
        top_5_mit = night_full_data['MIT_AST_label'].value_counts().head(5)
        for label, count in top_5_mit.items():
            pct = (count / len(night_full_data)) * 100
            summary['top_5_mit_labels'].append({
                'label': label,
                'count': int(count),
                'percentage': round(pct, 1)
            })
    
    return summary
