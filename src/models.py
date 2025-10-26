"""
Data models for the Image Artifact Detector application.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict, Any


@dataclass
class ArtifactResult:
    """Represents a detected artifact in an image."""
    artifact_type: str  # "moire" or "false_contour"
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # 0.0 to 1.0
    severity: str  # "low", "medium", "high"
    center_point: Tuple[int, int]
    area: int


@dataclass
class ImageInfo:
    """Contains basic information about an uploaded image."""
    filename: str
    width: int
    height: int
    channels: int
    file_size: int
    color_mode: str
    bit_depth: int


@dataclass
class DetectionReport:
    """Complete detection report for an analyzed image."""
    image_info: ImageInfo
    moire_patterns: List[ArtifactResult]
    false_contours: List[ArtifactResult]
    overall_score: float
    processing_time: float
    timestamp: datetime