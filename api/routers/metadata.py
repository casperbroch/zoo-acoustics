"""Metadata endpoints for zoos, enclosures, and available dates"""
from fastapi import APIRouter, HTTPException, Path
from typing import List
from datetime import date

from api.models import Zoo, Enclosure, AvailableDate, ErrorResponse, SpeciesInfo
from api.config import ZOOS_CONFIG
from api.services.data_processor import (
    load_zoo_enclosure_data,
    get_available_dates
)

router = APIRouter(prefix="/api", tags=["metadata"])


@router.get("/zoos", response_model=List[Zoo], summary="Get all zoos")
async def get_zoos():
    """
    Get a list of all available zoos with acoustic monitoring data.
    """
    zoos = []
    for zoo_name, zoo_data in ZOOS_CONFIG.items():
        zoos.append(Zoo(
            name=zoo_name,
            enclosure_count=len(zoo_data['enclosures'])
        ))
    return zoos


@router.get(
    "/zoos/{zoo_name}/enclosures",
    response_model=List[Enclosure],
    summary="Get enclosures for a zoo"
)
async def get_enclosures(
    zoo_name: str = Path(..., description="Name of the zoo")
):
    """
    Get all enclosures for a specific zoo.
    """
    if zoo_name not in ZOOS_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Zoo '{zoo_name}' not found. Available zoos: {list(ZOOS_CONFIG.keys())}"
        )
    
    enclosures = []
    zoo_data = ZOOS_CONFIG[zoo_name]
    
    for enclosure_name, enclosure_data in zoo_data['enclosures'].items():
        species_info = enclosure_data.get('species_info', {})
        
        enclosures.append(Enclosure(
            name=enclosure_name,
            zoo_name=zoo_name,
            species_count=len(species_info),
            species=[info['common'] for info in species_info.values()]
        ))
    
    return enclosures


@router.get(
    "/zoos/{zoo_name}/enclosures/{enclosure_name}/species",
    response_model=List[SpeciesInfo],
    summary="Get species information for an enclosure"
)
async def get_species_info(
    zoo_name: str = Path(..., description="Name of the zoo"),
    enclosure_name: str = Path(..., description="Name of the enclosure")
):
    """
    Get detailed information about target species in an enclosure.
    """
    if zoo_name not in ZOOS_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Zoo '{zoo_name}' not found"
        )
    
    if enclosure_name not in ZOOS_CONFIG[zoo_name]['enclosures']:
        raise HTTPException(
            status_code=404,
            detail=f"Enclosure '{enclosure_name}' not found in zoo '{zoo_name}'"
        )
    
    species_info = ZOOS_CONFIG[zoo_name]['enclosures'][enclosure_name].get('species_info', {})
    
    result = []
    for scientific_name, info in species_info.items():
        result.append(SpeciesInfo(
            scientific_name=scientific_name,
            common_name=info['common'],
            individual_count=info['count'],
            color=info['color']
        ))
    
    return result


@router.get(
    "/zoos/{zoo_name}/enclosures/{enclosure_name}/dates",
    response_model=List[AvailableDate],
    summary="Get available dates with data"
)
async def get_available_dates_endpoint(
    zoo_name: str = Path(..., description="Name of the zoo"),
    enclosure_name: str = Path(..., description="Name of the enclosure")
):
    """
    Get all available dates that have nighttime bird detection data.
    
    Each date represents a night starting at 8 PM on that date and ending at 3 AM the next day.
    """
    try:
        # Load data for this zoo/enclosure
        _, df_target = load_zoo_enclosure_data(zoo_name, enclosure_name)
        
        # Get available dates
        available_dates = get_available_dates(df_target)
        
        # Convert to response model
        result = [
            AvailableDate(
                date=d['date'].isoformat() if hasattr(d['date'], 'isoformat') else str(d['date']),
                detection_count=d['detection_count']
            )
            for d in available_dates
        ]
        
        return sorted(result, key=lambda x: x.date)
        
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Data files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
