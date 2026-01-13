"""Configuration for Zoo Acoustics API"""
from pathlib import Path
from typing import Dict

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Species information for GaiaZOO Congo enclosure
SPECIES_INFO = {
    'Ploceus melanocephalus': {
        'common': 'Black-headed weaver',
        'count': 5,
        'color': '#FFB300'
    },
    'Upupa epops': {
        'common': 'Eurasian hoopoe',
        'count': 2,
        'color': '#E91E63'
    },
    'Quelea quelea': {
        'common': 'Red-billed quelea',
        'count': 61,
        'color': '#2196F3'
    }
}

# Nighttime window definition (8 PM to 3 AM)
NIGHTTIME_HOURS = [20, 21, 22, 23, 0, 1, 2]

# Time binning configuration
TIME_BIN_MINUTES = 10

# Color gradient for activity visualization (black -> purple -> red)
ACTIVITY_COLORMAP = ['#000000', '#4a0e4e', '#8b008b', '#b8006b', '#dc143c', '#ff0000']

# Zoo and enclosure configuration
# For MVP: hardcoded single zoo/enclosure
ZOOS_CONFIG = {
    'GaiaZOO': {
        'enclosures': {
            'Congo': {
                'data_dir': 'fl_gaia_zoo_congo_15aug25_data',
                'full_data_file': 'fl_gaia_zoo_congo_15aug25_data__fl_gaia_zoo_congo_15aug25_data_metadata_speechless_predictions.xlsx',
                'target_data_file': 'target_species_detections.xlsx',
                'species_info': SPECIES_INFO
            }
        }
    }
}
