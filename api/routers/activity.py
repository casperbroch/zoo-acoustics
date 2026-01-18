"""Nighttime activity endpoint for visualization data"""
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import FileResponse
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from api.models import (
    NighttimeActivityResponse,
    SpeciesActivityData,
    OverallActivityData,
    MITSoundTypeData,
    SpeciesInfo,
    SpeciesActivityAudioResponse,
    MITSoundTypeAudioResponse,
    OverallActivityAudioResponse,
    OverallActivityAudioItem
)
from api.config import ZOOS_CONFIG
from api.config import DATA_DIR
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
        overall_counts = compute_overall_activity(night_full_data, time_bins)
        
        overall_activity_list = []
        for i, bin_center in enumerate(bin_centers):
            overall_activity_list.append(OverallActivityData(
                time=bin_center.strftime('%H:%M'),
                timestamp=bin_center.to_pydatetime(),
                total_clips=int(overall_counts[i])
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


@router.get(
    "/zoos/{zoo_name}/enclosures/{enclosure_name}/species-activity-audio",
    response_model=SpeciesActivityAudioResponse,
    summary="Get .wav clips for a species-activity time bin",
    description=(
        "Returns all bird-detection .wav files for the specified species-activity time bin. "
        "The `timestamp` must match a bin center from the `species_activity` graph (10-minute bins). "
        "Optionally filter to a specific species via `species`."
    )
)
async def get_species_activity_audio(
    zoo_name: str = Path(..., description="Name of the zoo"),
    enclosure_name: str = Path(..., description="Name of the enclosure"),
    timestamp: datetime = Query(
        ...,
        description="Bin center timestamp from species_activity (e.g. 2025-08-11T20:05:00)"
    ),
    species: Optional[str] = Query(
        None,
        description=(
            "Optional species filter (scientific names). Provide a single name or a comma-separated list, e.g. "
            "`species=Quelea quelea` or `species=Quelea quelea,Upupa epops`. If omitted, returns all species "
            "present in the bin."
        ),
        examples={
            "single": {"summary": "Single species", "value": "Quelea quelea"},
            "multiple": {"summary": "Two species", "value": "Quelea quelea,Upupa epops"}
        }
    ),
):
    """
    Get all .wav clips for the specified species-activity time bin.

    The `timestamp` must match a bin center returned by the `species_activity` graph.
    """
    try:
        # Load data
        df_full_clean, df_target = load_zoo_enclosure_data(zoo_name, enclosure_name)

        # Determine night_date from timestamp
        if timestamp.hour >= 20:
            night_date = timestamp.date()
        else:
            night_date = (timestamp - timedelta(days=1)).date()

        # Validate date exists
        if not validate_date_exists(night_date, df_target):
            available_dates = [d['date'] for d in get_available_dates(df_target)]
            raise HTTPException(
                status_code=404,
                detail=f"No nighttime bird detection data for {night_date}. Available dates: {sorted(available_dates)}"
            )

        # Filter data for nighttime window
        _, night_bird_data, night_start, night_end = filter_nighttime_data(
            night_date, df_full_clean, df_target
        )

        # Create time bins and find bin index for timestamp
        time_bins = create_time_bins(night_start, night_end)
        bin_centers = time_bins[:-1] + (time_bins[1] - time_bins[0]) / 2

        bin_index = None
        for i, bin_center in enumerate(bin_centers):
            if bin_center.to_pydatetime() == timestamp:
                bin_index = i
                break

        if bin_index is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Timestamp must match a species_activity bin center (10-minute bins). "
                    "Use a timestamp returned by the activity endpoint."
                )
            )

        bin_start = time_bins[bin_index]
        bin_end = time_bins[bin_index + 1]

        # Optional species filter
        species_info_dict = ZOOS_CONFIG[zoo_name]['enclosures'][enclosure_name]['species_info']
        requested_species: List[str] = []
        filtered_bird_data = night_bird_data
        if species:
            requested_species = [s.strip() for s in species.split(',') if s.strip()]
            requested_species = list(dict.fromkeys(requested_species))
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
            filtered_bird_data = night_bird_data[
                night_bird_data['detected_species'].isin(requested_species)
            ].copy()

        # Filter for the specified bin
        bin_data = filtered_bird_data[
            (filtered_bird_data['datetime'] >= bin_start) &
            (filtered_bird_data['datetime'] < bin_end)
        ].copy()

        species_audio: Dict[str, List[str]] = {}
        if not bin_data.empty:
            for species_name, group in bin_data.groupby('detected_species'):
                files = group['filename'].dropna().astype(str).tolist()
                if files:
                    species_audio[species_name] = files

        return SpeciesActivityAudioResponse(
            zoo_name=zoo_name,
            enclosure_name=enclosure_name,
            night_date=night_date,
            timestamp=timestamp,
            bin_start=bin_start.to_pydatetime(),
            bin_end=bin_end.to_pydatetime(),
            species_audio=species_audio,
            total_clips=int(len(bin_data))
        )

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@router.get(
    "/zoos/{zoo_name}/enclosures/{enclosure_name}/audio",
    summary="Get a .wav file by path",
    description=(
        "Returns the .wav file for a given path returned by the species-activity audio endpoint. "
        "Provide the path exactly as returned (relative to the data root)."
    )
)
async def get_audio_file(
    zoo_name: str = Path(..., description="Name of the zoo"),
    enclosure_name: str = Path(..., description="Name of the enclosure"),
    file_path: str = Query(
        ...,
        description="Relative .wav path returned by species-activity audio endpoint"
    )
):
    """Return a .wav audio file by relative path."""
    try:
        if zoo_name not in ZOOS_CONFIG:
            raise HTTPException(status_code=404, detail=f"Zoo '{zoo_name}' not found in configuration")
        if enclosure_name not in ZOOS_CONFIG[zoo_name]['enclosures']:
            raise HTTPException(
                status_code=404,
                detail=f"Enclosure '{enclosure_name}' not found for zoo '{zoo_name}'"
            )

        if not file_path or not file_path.lower().endswith('.wav'):
            raise HTTPException(status_code=400, detail="file_path must point to a .wav file")

        config = ZOOS_CONFIG[zoo_name]['enclosures'][enclosure_name]
        data_dir_name = config['data_dir']

        # Resolve candidate path against data root
        candidate = (DATA_DIR / file_path).resolve()
        data_root = DATA_DIR.resolve()

        # Ensure path is within data root and within enclosure data directory
        if data_root not in candidate.parents and candidate != data_root:
            raise HTTPException(status_code=400, detail="Invalid file_path (outside data directory)")

        enclosure_root = (DATA_DIR / data_dir_name).resolve()
        if enclosure_root not in candidate.parents and candidate != enclosure_root:
            raise HTTPException(status_code=400, detail="Invalid file_path (outside enclosure data directory)")

        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(
            path=str(candidate),
            media_type="audio/wav",
            filename=candidate.name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading audio file: {str(e)}")


@router.get(
    "/zoos/{zoo_name}/enclosures/{enclosure_name}/mit-audio",
    response_model=MITSoundTypeAudioResponse,
    summary="Get .wav paths for a MIT sound-type time bin",
    description=(
        "Returns all .wav file paths for a given MIT sound type within the specified time bin. "
        "The `timestamp` must match a bin center from the `mit_breakdown` graph (10-minute bins)."
    )
)
async def get_mit_sound_type_audio(
    zoo_name: str = Path(..., description="Name of the zoo"),
    enclosure_name: str = Path(..., description="Name of the enclosure"),
    timestamp: datetime = Query(
        ...,
        description="Bin center timestamp from mit_breakdown (e.g. 2025-08-11T20:05:00)"
    ),
    sound_type: str = Query(
        ...,
        description="MIT sound type label (e.g. rain, wind, insect)"
    )
):
    """Get .wav paths for a MIT sound type in the specified time bin."""
    try:
        # Load data
        df_full_clean, df_target = load_zoo_enclosure_data(zoo_name, enclosure_name)

        # Determine night_date from timestamp
        if timestamp.hour >= 20:
            night_date = timestamp.date()
        else:
            night_date = (timestamp - timedelta(days=1)).date()

        # Validate date exists
        if not validate_date_exists(night_date, df_target):
            available_dates = [d['date'] for d in get_available_dates(df_target)]
            raise HTTPException(
                status_code=404,
                detail=f"No nighttime data for {night_date}. Available dates: {sorted(available_dates)}"
            )

        # Filter data for nighttime window
        night_full_data, _, night_start, night_end = filter_nighttime_data(
            night_date, df_full_clean, df_target
        )

        # Create time bins and find bin index for timestamp
        time_bins = create_time_bins(night_start, night_end)
        bin_centers = time_bins[:-1] + (time_bins[1] - time_bins[0]) / 2

        bin_index = None
        for i, bin_center in enumerate(bin_centers):
            if bin_center.to_pydatetime() == timestamp:
                bin_index = i
                break

        if bin_index is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Timestamp must match a mit_breakdown bin center (10-minute bins). "
                    "Use a timestamp returned by the activity endpoint."
                )
            )

        if not sound_type or not sound_type.strip():
            raise HTTPException(status_code=400, detail="sound_type is required")

        sound_type = sound_type.strip()

        bin_start = time_bins[bin_index]
        bin_end = time_bins[bin_index + 1]

        # Filter for the specified bin and sound type
        bin_data = night_full_data[
            (night_full_data['datetime'] >= bin_start) &
            (night_full_data['datetime'] < bin_end)
        ]

        bin_data = bin_data[bin_data['MIT_AST_label'] == sound_type]

        audio_paths = []
        if not bin_data.empty and 'filename' in bin_data.columns:
            audio_paths = bin_data['filename'].dropna().astype(str).tolist()

        return MITSoundTypeAudioResponse(
            zoo_name=zoo_name,
            enclosure_name=enclosure_name,
            night_date=night_date,
            timestamp=timestamp,
            bin_start=bin_start.to_pydatetime(),
            bin_end=bin_end.to_pydatetime(),
            sound_type=sound_type,
            audio_paths=audio_paths,
            total_clips=int(len(audio_paths))
        )

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@router.get(
    "/zoos/{zoo_name}/enclosures/{enclosure_name}/overall-audio",
    response_model=OverallActivityAudioResponse,
    summary="Get .wav paths for an overall-activity time bin",
    description=(
        "Returns all .wav file paths for the specified overall-activity time bin. "
        "The `timestamp` must match a bin center from the `overall_activity` graph (10-minute bins). "
        "Each returned item includes its MIT sound type."
    )
)
async def get_overall_activity_audio(
    zoo_name: str = Path(..., description="Name of the zoo"),
    enclosure_name: str = Path(..., description="Name of the enclosure"),
    timestamp: datetime = Query(
        ...,
        description="Bin center timestamp from overall_activity (e.g. 2025-08-11T20:05:00)"
    )
):
    """Get .wav paths for overall activity in the specified time bin."""
    try:
        # Load data
        df_full_clean, df_target = load_zoo_enclosure_data(zoo_name, enclosure_name)

        # Determine night_date from timestamp
        if timestamp.hour >= 20:
            night_date = timestamp.date()
        else:
            night_date = (timestamp - timedelta(days=1)).date()

        # Validate date exists
        if not validate_date_exists(night_date, df_target):
            available_dates = [d['date'] for d in get_available_dates(df_target)]
            raise HTTPException(
                status_code=404,
                detail=f"No nighttime data for {night_date}. Available dates: {sorted(available_dates)}"
            )

        # Filter data for nighttime window
        night_full_data, _, night_start, night_end = filter_nighttime_data(
            night_date, df_full_clean, df_target
        )

        # Create time bins and find bin index for timestamp
        time_bins = create_time_bins(night_start, night_end)
        bin_centers = time_bins[:-1] + (time_bins[1] - time_bins[0]) / 2

        bin_index = None
        for i, bin_center in enumerate(bin_centers):
            if bin_center.to_pydatetime() == timestamp:
                bin_index = i
                break

        if bin_index is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Timestamp must match an overall_activity bin center (10-minute bins). "
                    "Use a timestamp returned by the activity endpoint."
                )
            )

        bin_start = time_bins[bin_index]
        bin_end = time_bins[bin_index + 1]

        # Filter for the specified bin
        bin_data = night_full_data[
            (night_full_data['datetime'] >= bin_start) &
            (night_full_data['datetime'] < bin_end)
        ]

        audio_items = []
        if not bin_data.empty and 'filename' in bin_data.columns:
            for _, row in bin_data.iterrows():
                file_path = row.get('filename')
                if file_path is None:
                    continue
                file_path = str(file_path)
                if not file_path:
                    continue
                audio_items.append(OverallActivityAudioItem(
                    file_path=file_path,
                    sound_type=str(row.get('MIT_AST_label', ''))
                ))

        return OverallActivityAudioResponse(
            zoo_name=zoo_name,
            enclosure_name=enclosure_name,
            night_date=night_date,
            timestamp=timestamp,
            bin_start=bin_start.to_pydatetime(),
            bin_end=bin_end.to_pydatetime(),
            audio=audio_items,
            total_clips=int(len(audio_items))
        )

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
