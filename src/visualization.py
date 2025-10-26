"""
Visualization module for artifact detection results.
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any
from src.models import ArtifactResult


class ArtifactVisualizer:
    """Handles visualization of detected artifacts with color coding."""
    
    # Color definitions (BGR format for OpenCV)
    MOIRE_COLOR = (0, 0, 255)      # Red for Moiré patterns
    CONTOUR_COLOR = (255, 0, 0)    # Blue for False contours
    
    # RGB format for PIL
    MOIRE_COLOR_RGB = (255, 0, 0)      # Red for Moiré patterns
    CONTOUR_COLOR_RGB = (0, 0, 255)    # Blue for False contours
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def draw_artifact_annotations(self, image: np.ndarray, artifacts: List[ArtifactResult]) -> np.ndarray:
        """
        Draw semi-transparent overlays and numbered markers for detected artifacts.
        
        Args:
            image: Original image array (RGB)
            artifacts: List of detected artifacts
            
        Returns:
            Annotated image array with semi-transparent overlays
        """
        # Create a copy of the image
        annotated = image.copy().astype(np.float32)
        
        # Create overlay masks for different artifact types
        moire_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Separate artifacts by type for numbering
        moire_count = 0
        contour_count = 0
        center_points = []  # Store center points for numbering
        
        for artifact in artifacts:
            x, y, w, h = artifact.bounding_box
            center_x, center_y = artifact.center_point
            
            if artifact.artifact_type == "moire":
                # Fill the bounding box area in moire mask
                moire_mask[y:y+h, x:x+w] = 255
                moire_count += 1
                center_points.append((center_x, center_y, "moire", moire_count))
            else:  # false_contour
                # Fill the bounding box area in contour mask
                contour_mask[y:y+h, x:x+w] = 255
                contour_count += 1
                center_points.append((center_x, center_y, "contour", contour_count))
        
        # Apply semi-transparent overlays
        alpha = 0.3  # Transparency level (0.3 = 30% overlay, 70% original)
        
        # Apply red overlay for Moiré patterns
        if np.any(moire_mask > 0):
            red_overlay = np.array(self.MOIRE_COLOR_RGB, dtype=np.float32)
            for i in range(3):  # RGB channels
                annotated[:, :, i] = np.where(
                    moire_mask > 0,
                    annotated[:, :, i] * (1 - alpha) + red_overlay[i] * alpha,
                    annotated[:, :, i]
                )
        
        # Apply blue overlay for False contours
        if np.any(contour_mask > 0):
            blue_overlay = np.array(self.CONTOUR_COLOR_RGB, dtype=np.float32)
            for i in range(3):  # RGB channels
                annotated[:, :, i] = np.where(
                    contour_mask > 0,
                    annotated[:, :, i] * (1 - alpha) + blue_overlay[i] * alpha,
                    annotated[:, :, i]
                )
        
        # Convert back to uint8
        annotated = np.clip(annotated, 0, 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV operations (for drawing center points and numbers)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        # Draw center points and numbers
        for center_x, center_y, artifact_type, number in center_points:
            if artifact_type == "moire":
                color = self.MOIRE_COLOR
            else:
                color = self.CONTOUR_COLOR
            
            # Draw center point (smaller, more subtle)
            cv2.circle(annotated_bgr, (center_x, center_y), 3, color, -1)
            cv2.circle(annotated_bgr, (center_x, center_y), 4, (255, 255, 255), 1)  # White border
            
            # Draw number near the center point
            number_text = str(number)
            
            # Calculate text position (offset from center)
            text_x = center_x + 8
            text_y = center_y - 8
            
            # Ensure text stays within image bounds
            text_x = max(15, min(text_x, annotated_bgr.shape[1] - 20))
            text_y = max(20, min(text_y, annotated_bgr.shape[0] - 5))
            
            # Draw number with smaller background circle
            cv2.circle(annotated_bgr, (text_x, text_y), 10, color, -1)
            cv2.circle(annotated_bgr, (text_x, text_y), 11, (255, 255, 255), 1)  # White border
            
            # Draw number text
            cv2.putText(annotated_bgr, number_text, (text_x - 4, text_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Convert back to RGB
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        return annotated_rgb
    
    def create_comparison_view(self, original: np.ndarray, annotated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create side-by-side comparison of original and annotated images.
        
        Args:
            original: Original image array
            annotated: Annotated image array
            
        Returns:
            Tuple of (original, annotated) images resized for comparison
        """
        # Ensure both images have the same dimensions
        if original.shape != annotated.shape:
            # Resize annotated to match original
            if len(original.shape) == 3:
                h, w = original.shape[:2]
            else:
                h, w = original.shape
            annotated = cv2.resize(annotated, (w, h))
        
        return original, annotated
    
    def generate_heatmap(self, confidence_map: np.ndarray, artifact_type: str = "moire") -> np.ndarray:
        """
        Generate a heatmap visualization of detection confidence.
        
        Args:
            confidence_map: 2D array of confidence values
            artifact_type: Type of artifact for color selection
            
        Returns:
            Heatmap image array
        """
        # Normalize confidence map to 0-255
        normalized = (confidence_map * 255).astype(np.uint8)
        
        # Apply colormap
        if artifact_type == "moire":
            # Red colormap for Moiré patterns
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
        else:
            # Blue colormap for False contours
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_WINTER)
        
        # Convert to RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap_rgb
    
    def create_overlay_visualization(self, original: np.ndarray, artifacts: List[ArtifactResult], 
                                   alpha: float = 0.3) -> np.ndarray:
        """
        Create an overlay visualization with semi-transparent artifact regions.
        
        Args:
            original: Original image array
            artifacts: List of detected artifacts
            alpha: Transparency level for overlay
            
        Returns:
            Overlay visualization
        """
        # Create overlay image
        overlay = original.copy()
        
        # Create masks for different artifact types
        moire_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        contour_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        
        for artifact in artifacts:
            x, y, w, h = artifact.bounding_box
            
            if artifact.artifact_type == "moire":
                moire_mask[y:y+h, x:x+w] = 255
            else:
                contour_mask[y:y+h, x:x+w] = 255
        
        # Apply colored overlays
        if len(original.shape) == 3:
            # Color image
            overlay_colored = original.copy()
            
            # Red overlay for Moiré patterns
            overlay_colored[moire_mask > 0] = (
                overlay_colored[moire_mask > 0] * (1 - alpha) + 
                np.array(self.MOIRE_COLOR_RGB) * alpha
            ).astype(np.uint8)
            
            # Blue overlay for False contours
            overlay_colored[contour_mask > 0] = (
                overlay_colored[contour_mask > 0] * (1 - alpha) + 
                np.array(self.CONTOUR_COLOR_RGB) * alpha
            ).astype(np.uint8)
        else:
            # Grayscale image - convert to RGB first
            overlay_colored = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            
            # Apply overlays
            overlay_colored[moire_mask > 0] = (
                overlay_colored[moire_mask > 0] * (1 - alpha) + 
                np.array(self.MOIRE_COLOR_RGB) * alpha
            ).astype(np.uint8)
            
            overlay_colored[contour_mask > 0] = (
                overlay_colored[contour_mask > 0] * (1 - alpha) + 
                np.array(self.CONTOUR_COLOR_RGB) * alpha
            ).astype(np.uint8)
        
        return overlay_colored
    
    def create_legend(self, width: int = 200, height: int = 100) -> np.ndarray:
        """
        Create a legend showing artifact type colors.
        
        Args:
            width: Legend width
            height: Legend height
            
        Returns:
            Legend image array
        """
        # Create white background
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw color boxes and labels
        box_size = 20
        y_offset = 20
        
        # Moiré pattern legend
        cv2.rectangle(legend, (10, y_offset), (10 + box_size, y_offset + box_size), 
                     self.MOIRE_COLOR[::-1], -1)  # Convert BGR to RGB
        cv2.putText(legend, "Moire Pattern", (40, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # False contour legend
        y_offset += 40
        cv2.rectangle(legend, (10, y_offset), (10 + box_size, y_offset + box_size), 
                     self.CONTOUR_COLOR[::-1], -1)  # Convert BGR to RGB
        cv2.putText(legend, "False Contour", (40, y_offset + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return legend
    
    def filter_artifacts_by_type(self, artifacts: List[ArtifactResult], 
                                artifact_type: str) -> List[ArtifactResult]:
        """
        Filter artifacts by type for selective visualization.
        
        Args:
            artifacts: List of all artifacts
            artifact_type: Type to filter ("moire", "false_contour", or "all")
            
        Returns:
            Filtered list of artifacts
        """
        if artifact_type == "all":
            return artifacts
        
        return [artifact for artifact in artifacts if artifact.artifact_type == artifact_type]
    
    def create_detailed_annotation(self, image: np.ndarray, artifact: ArtifactResult) -> np.ndarray:
        """
        Create detailed annotation for a single artifact.
        
        Args:
            image: Original image array
            artifact: Single artifact to annotate
            
        Returns:
            Detailed annotation image
        """
        annotated = image.copy()
        
        # Get color and label
        if artifact.artifact_type == "moire":
            color = self.MOIRE_COLOR_RGB
            type_label = "Moire Pattern"
        else:
            color = self.CONTOUR_COLOR_RGB
            type_label = "False Contour"
        
        # Convert to PIL for better text rendering
        pil_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(pil_image)
        
        # Draw bounding box
        x, y, w, h = artifact.bounding_box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        # Draw center point
        center_x, center_y = artifact.center_point
        draw.ellipse([center_x - 3, center_y - 3, center_x + 3, center_y + 3], 
                    fill=color)
        
        # Create detailed label
        details = [
            type_label,
            f"Confidence: {artifact.confidence:.3f}",
            f"Severity: {artifact.severity}",
            f"Area: {artifact.area} pixels",
            f"Center: ({center_x}, {center_y})"
        ]
        
        # Draw label background and text
        try:
            # Try to use a better font if available
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        label_y = max(10, y - len(details) * 15 - 10)
        for i, detail in enumerate(details):
            draw.text((x, label_y + i * 15), detail, fill=color, font=font)
        
        return np.array(pil_image)