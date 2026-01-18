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


class SpeciesActivityAudioResponse(BaseModel):
    """Audio clips for a species activity time bin"""
    zoo_name: str
    enclosure_name: str
    night_date: date = Field(
        ...,
        description="The date of the night (starts at 8 PM this date, ends 3 AM next day)"
    )
    timestamp: datetime = Field(
        ...,
        description="Exact timestamp of the bin center (must match species_activity timestamps)"
    )
    bin_start: datetime = Field(..., description="Start timestamp of the bin (inclusive)")
    bin_end: datetime = Field(..., description="End timestamp of the bin (exclusive)")
    species_audio: Dict[str, List[str]] = Field(
        ...,
        description="Mapping of species (scientific name) to list of .wav file paths"
    )
    total_clips: int = Field(..., description="Total number of bird detections in this bin")


class MITSoundTypeAudioResponse(BaseModel):
    """Audio clips for a MIT sound-type time bin"""
    zoo_name: str
    enclosure_name: str
    night_date: date = Field(
        ...,
        description="The date of the night (starts at 8 PM this date, ends 3 AM next day)"
    )
    timestamp: datetime = Field(
        ...,
        description="Exact timestamp of the bin center (must match mit_breakdown timestamps)"
    )
    bin_start: datetime = Field(..., description="Start timestamp of the bin (inclusive)")
    bin_end: datetime = Field(..., description="End timestamp of the bin (exclusive)")
    sound_type: str = Field(..., description="MIT sound type label")
    audio_paths: List[str] = Field(..., description="List of .wav file paths")
    total_clips: int = Field(..., description="Total number of clips in this bin for the sound type")


class OverallActivityAudioItem(BaseModel):
    """Audio item with MIT sound type"""
    file_path: str = Field(..., description=".wav file path")
    sound_type: str = Field(..., description="MIT sound type label")


class OverallActivityAudioResponse(BaseModel):
    """Audio clips for overall activity time bin"""
    zoo_name: str
    enclosure_name: str
    night_date: date = Field(
        ...,
        description="The date of the night (starts at 8 PM this date, ends 3 AM next day)"
    )
    timestamp: datetime = Field(
        ...,
        description="Exact timestamp of the bin center (must match overall_activity timestamps)"
    )
    bin_start: datetime = Field(..., description="Start timestamp of the bin (inclusive)")
    bin_end: datetime = Field(..., description="End timestamp of the bin (exclusive)")
    audio: List[OverallActivityAudioItem] = Field(
        ...,
        description="List of audio paths with their MIT sound types"
    )
    total_clips: int = Field(..., description="Total number of clips in this bin")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
