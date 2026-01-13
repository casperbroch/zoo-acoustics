"""Pydantic models for API request/response schemas"""
from datetime import date, datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class Zoo(BaseModel):
    """Zoo information"""
    name: str
    enclosure_count: int


class Enclosure(BaseModel):
    """Enclosure information"""
    name: str
    zoo_name: str
    species_count: int
    species: List[str]


class AvailableDate(BaseModel):
    """Available date with data count"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    detection_count: int


class SpeciesActivityData(BaseModel):
    """Bird species activity data for a single time bin"""
    time: str = Field(..., description="Time label (HH:MM format)")
    timestamp: datetime = Field(..., description="Exact timestamp of bin center")
    species_counts: Dict[str, int] = Field(
        ...,
        description="Counts per species (scientific name -> count)"
    )


class OverallActivityData(BaseModel):
    """Overall acoustic activity for a single time bin"""
    time: str = Field(..., description="Time label (HH:MM format)")
    timestamp: datetime = Field(..., description="Exact timestamp of bin center")
    total_clips: int = Field(..., description="Total sound clips in this bin")
    intensity: float = Field(..., description="Normalized intensity (0-1)")


class MITSoundTypeData(BaseModel):
    """MIT sound type data for a single time bin"""
    time: str = Field(..., description="Time label (HH:MM format)")
    timestamp: datetime = Field(..., description="Exact timestamp of bin center")
    sound_type_counts: Dict[str, int] = Field(
        ...,
        description="Counts per MIT sound type"
    )


class SpeciesInfo(BaseModel):
    """Species information"""
    scientific_name: str
    common_name: str
    individual_count: int
    color: str


class NighttimeActivityResponse(BaseModel):
    """Complete nighttime activity data for visualization"""
    zoo_name: str
    enclosure_name: str
    night_date: date = Field(
        ...,
        description="The date of the night (starts at 8 PM this date, ends 3 AM next day)"
    )
    night_start: datetime
    night_end: datetime
    
    species_activity: List[SpeciesActivityData] = Field(
        ...,
        description="Target species bird detections per 10-minute bin"
    )
    
    overall_activity: List[OverallActivityData] = Field(
        ...,
        description="Overall acoustic activity (all sounds) per 10-minute bin"
    )
    
    mit_breakdown: List[MITSoundTypeData] = Field(
        ...,
        description="MIT sound classification breakdown per 10-minute bin"
    )
    
    species_info: List[SpeciesInfo] = Field(
        ...,
        description="Information about detected species"
    )
    
    top_mit_labels: List[str] = Field(
        ...,
        description="Top 8 MIT classification labels for this night"
    )
    
    midnight_index: Optional[int] = Field(
        None,
        description="Index of the time bin containing midnight (for visualization)"
    )
    
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics for the night"
    )


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
