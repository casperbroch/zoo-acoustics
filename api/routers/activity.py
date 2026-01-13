"""Nighttime activity endpoint for visualization data"""
from fastapi import APIRouter, HTTPException, Path, Query
from datetime import date, datetime
from typing import Dict, List, Optional

from api.models import (
    NighttimeActivityResponse,
    SpeciesActivityData,
    OverallActivityData,
    MITSoundTypeData,
    SpeciesInfo
)
from api.config import ZOOS_CONFIG
from api.services.data_processor import (
    load_zoo_enclosure_data,
    get_available_dates,
    validate_date_exists,
    filter_nighttime_data,
    create_time_bins,
    compute_species_activity,
    compute_overall_activity,
    compute_mit_breakdown,
    find_midnight_index,
    compute_summary_statistics
)

router = APIRouter(prefix="/api", tags=["activity"])


@router.get(
    "/zoos/{zoo_name}/enclosures/{enclosure_name}/activity",
    response_model=NighttimeActivityResponse,
    summary="Get nighttime activity data for visualization",
    description=(
        "Returns nighttime (8 PM → 3 AM) acoustic activity binned into 10-minute intervals. "
        "Optionally filter the bird-species activity (and bird-related summary stats) to one or more species.\n\n"
        "Notes:\n"
        "- `species` filtering only affects the bird detections view (species activity + bird summary).\n"
        "- Overall activity (all sounds) and MIT breakdown remain for the full night window."
    )
)
async def get_nighttime_activity(
    zoo_name: str = Path(..., description="Name of the zoo"),
    enclosure_name: str = Path(..., description="Name of the enclosure"),
    date: date = Query(
        ...,
        description="The night date (YYYY-MM-DD). Night window is 8 PM this date to 3 AM next day."
    ),
    species: Optional[str] = Query(
        None,
        description=(
            "Optional species filter (scientific names). Provide a single name or a comma-separated list, e.g. "
            "`species=Quelea quelea` or `species=Quelea quelea,Upupa epops`. If omitted, returns activity for all "
            "detected target species."
        ),
        examples={
            "single": {"summary": "Single species", "value": "Quelea quelea"},
            "multiple": {"summary": "Two species", "value": "Quelea quelea,Upupa epops"}
        }
    ),
):
    """
    Get complete nighttime acoustic activity data for a specific date.
    
    Returns three datasets ready for frontend visualization:
    1. **Species Activity**: Bird detections per species per 10-minute bin
    2. **Overall Activity**: Total acoustic activity (all sounds) per 10-minute bin
    3. **MIT Breakdown**: Sound type classification breakdown per 10-minute bin
    
    The night window spans from 8 PM on the specified date to 3 AM the following day.
    
    **Example:**
    - `date=2024-08-10` → Night of August 10 (8 PM Aug 10 to 3 AM Aug 11)
    - `date=2024-08-09` → Night of August 9 (8 PM Aug 9 to 3 AM Aug 10)

    **Filter by species (comma-separated):**
    - `date=2024-08-10&species=Quelea quelea`
    - `date=2024-08-10&species=Quelea quelea,Upupa epops`
    """
    try:
        # Load data
        df_full_clean, df_target = load_zoo_enclosure_data(zoo_name, enclosure_name)
        
        # Validate date exists
        if not validate_date_exists(date, df_target):
            available_dates = [d['date'] for d in get_available_dates(df_target)]
            raise HTTPException(
                status_code=404,
                detail=f"No nighttime bird detection data for {date}. Available dates: {sorted(available_dates)}"
            )
        
        # Filter data for nighttime window
        night_full_data, night_bird_data, night_start, night_end = filter_nighttime_data(
            date, df_full_clean, df_target
        )
        
        # Create time bins
        time_bins = create_time_bins(night_start, night_end)
        bin_centers = time_bins[:-1] + (time_bins[1] - time_bins[0]) / 2
        
        species_info_dict = ZOOS_CONFIG[zoo_name]['enclosures'][enclosure_name]['species_info']

        # Optionally filter bird detections by species
        requested_species: List[str] = []
        night_bird_data_filtered = night_bird_data
        if species:
            requested_species = [s.strip() for s in species.split(',') if s.strip()]
            requested_species = list(dict.fromkeys(requested_species))  # de-dupe while preserving order
            if not requested_species:
                raise HTTPException(
                    status_code=400,
                    detail="Species filter is empty after parsing; provide at least one scientific name."
                )
            unknown = [s for s in requested_species if s not in species_info_dict]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unknown species for this enclosure: "
                        f"{unknown}. Available: {sorted(list(species_info_dict.keys()))}"
                    )
                )
            night_bird_data_filtered = night_bird_data[
                night_bird_data['detected_species'].isin(requested_species)
            ].copy()

        # Compute species activity
        bird_activity = compute_species_activity(night_bird_data_filtered, time_bins, species_info_dict)
        
        # Build species activity response
        species_activity_list = []
        for i, bin_center in enumerate(bin_centers):
            species_counts = {}
            for species, counts in bird_activity.items():
                if counts[i] > 0:  # Only include non-zero counts
                    species_counts[species] = int(counts[i])
            
            species_activity_list.append(SpeciesActivityData(
                time=bin_center.strftime('%H:%M'),
                timestamp=bin_center.to_pydatetime(),
                species_counts=species_counts
            ))
        
        # Compute overall activity
        overall_counts, intensities = compute_overall_activity(night_full_data, time_bins)
        
        overall_activity_list = []
        for i, bin_center in enumerate(bin_centers):
            overall_activity_list.append(OverallActivityData(
                time=bin_center.strftime('%H:%M'),
                timestamp=bin_center.to_pydatetime(),
                total_clips=int(overall_counts[i]),
                intensity=float(intensities[i])
            ))
        
        # Compute MIT breakdown
        top_mit_labels, mit_categories = compute_mit_breakdown(night_full_data, time_bins)
        
        mit_breakdown_list = []
        for i, bin_center in enumerate(bin_centers):
            sound_type_counts = {}
            if i in mit_categories:
                # Only include top MIT labels
                for label in top_mit_labels:
                    count = mit_categories[i].get(label, 0)
                    if count > 0:
                        sound_type_counts[label] = count
            
            mit_breakdown_list.append(MITSoundTypeData(
                time=bin_center.strftime('%H:%M'),
                timestamp=bin_center.to_pydatetime(),
                sound_type_counts=sound_type_counts
            ))
        
        # Find midnight index
        midnight_idx = find_midnight_index(time_bins)
        
        # Compute summary statistics
        summary = compute_summary_statistics(night_bird_data_filtered, night_full_data, species_info_dict)
        
        # Build species info list
        species_info_list = []
        if species:
            species_to_describe = requested_species
        else:
            species_to_describe = night_bird_data_filtered['detected_species'].unique().tolist()

        for sp in species_to_describe:
            if sp not in species_info_dict:
                continue
            info = species_info_dict[sp]
            species_info_list.append(SpeciesInfo(
                scientific_name=sp,
                common_name=info['common'],
                individual_count=info['count'],
                color=info['color']
            ))
        
        # Build final response
        return NighttimeActivityResponse(
            zoo_name=zoo_name,
            enclosure_name=enclosure_name,
            night_date=date,
            night_start=night_start.to_pydatetime(),
            night_end=night_end.to_pydatetime(),
            species_activity=species_activity_list,
            overall_activity=overall_activity_list,
            mit_breakdown=mit_breakdown_list,
            species_info=species_info_list,
            top_mit_labels=top_mit_labels,
            midnight_index=midnight_idx,
            summary=summary
        )
        
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
