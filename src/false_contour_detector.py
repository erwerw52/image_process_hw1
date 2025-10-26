"""
假輪廓檢測模組 - 使用梯度分析和邊緣檢測技術檢測影像中的假輪廓偽影
False Contour detection module using gradient analysis and edge detection.

假輪廓是由於量化不足導致的階梯狀邊緣效應，
通常出現在色彩漸變區域，表現為不自然的階梯狀邊界。
"""
import numpy as np
import cv2
from scipy import ndimage
from typing import List, Dict, Any, Tuple
from src.models import ArtifactResult


class FalseContourDetector:
    """
    假輪廓檢測器 - 使用梯度分析檢測影像中的假輪廓
    Detects false contours in images using gradient analysis.
    
    檢測原理：
    1. 分析影像梯度的不連續性
    2. 檢測階梯狀邊緣模式
    3. 評估紋理一致性
    4. 識別假輪廓區域並計算信心度
    """
    
    def __init__(self, min_confidence: float = 0.3):
        """
        初始化假輪廓檢測器
        Initialize the false contour detector.
        
        Args:
            min_confidence: 最小信心度閾值，低於此值的檢測結果將被過濾
                          Minimum confidence threshold for detection
        """
        self.min_confidence = min_confidence
    
    def detect_false_contours(self, image: np.ndarray) -> List[ArtifactResult]:
        """
        檢測輸入影像中的假輪廓
        Detect false contours in the input image.
        
        Args:
            image: 輸入影像陣列（灰階或彩色）
                  Input image array (grayscale or RGB)
            
        Returns:
            檢測到的假輪廓偽影列表
            List of detected false contour artifacts
        """
        # 步驟1：轉換為灰階影像（如果需要）
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # 步驟2：調整大圖尺寸以提升處理速度
        # Resize large images for faster processing
        original_shape = gray_image.shape
        max_size = 512  # 限制處理尺寸以提升速度
        
        if max(original_shape) > max_size:
            # 計算縮放比例，保持長寬比
            scale_factor = max_size / max(original_shape)
            new_height = int(original_shape[0] * scale_factor)
            new_width = int(original_shape[1] * scale_factor)
            gray_image = cv2.resize(gray_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            scale_factor = 1.0
        
        # 步驟3：正規化影像數值到 0-1 範圍
        # Normalize image
        gray_image = gray_image.astype(np.float32) / 255.0
        
        # 步驟4：使用快速檢測方法
        # Use fast detection method
        contour_regions = self._fast_contour_detection(gray_image)
        
        # 步驟5：如果影像被縮放，將座標縮放回原始尺寸
        # Scale back coordinates if image was resized
        if scale_factor != 1.0:
            for region in contour_regions:
                x, y, w, h = region['bbox']
                # 縮放邊界框座標
                region['bbox'] = (
                    int(x / scale_factor),
                    int(y / scale_factor),
                    int(w / scale_factor),
                    int(h / scale_factor)
                )
                # 縮放中心點座標
                cx, cy = region['center']
                region['center'] = (
                    int(cx / scale_factor),
                    int(cy / scale_factor)
                )
                # 縮放面積
                region['area'] = int(region['area'] / (scale_factor * scale_factor))
        
        # 步驟6：轉換為 ArtifactResult 物件
        # Convert regions to ArtifactResult objects
        artifacts = []
        for region in contour_regions:
            artifact = ArtifactResult(
                artifact_type="false_contour",
                bounding_box=region['bbox'],
                confidence=region['confidence'],
                severity=self._calculate_severity(region['confidence']),
                center_point=region['center'],
                area=region['area']
            )
            artifacts.append(artifact)
        
        return artifacts
    
    def _fast_contour_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        快速假輪廓檢測 - 使用簡化梯度分析和輪廓檢測
        Fast false contour detection using simplified gradient analysis.
        
        檢測策略：
        1. 計算影像梯度以識別邊緣
        2. 使用 Canny 邊緣檢測找出邊界
        3. 通過形態學操作檢測階梯狀模式
        4. 分析邊緣密度和梯度變化特徵
        
        Args:
            image: 灰階影像陣列
                  Grayscale image array
            
        Returns:
            檢測到的區域列表
            List of detected regions
        """
        regions = []
        
        # 步驟1：快速梯度分析
        # Quick gradient analysis
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)  # X方向梯度
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)  # Y方向梯度
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)  # 梯度幅度
        
        # 步驟2：簡單邊緣檢測
        # Simple edge detection
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        
        # 步驟3：尋找階梯狀模式（假輪廓的特徵）
        # Look for step-like patterns
        # 水平和垂直形態學核心，用於檢測階梯狀邊緣
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # 水平核心
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # 垂直核心
        
        # 檢測水平和垂直階梯模式
        step_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)  # 水平階梯
        step_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)  # 垂直階梯
        step_edges = np.maximum(step_h, step_v)  # 合併階梯邊緣
        
        # 步驟4：在階梯邊緣中尋找輪廓
        # Find contours in step edges
        contours, _ = cv2.findContours(step_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 步驟5：分析每個輪廓
        for contour in contours:
            # 根據面積過濾小區域
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 100:  # 忽略太小的區域
                continue
            
            # 獲取邊界框
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # 步驟6：快速信心度檢查
            # Quick confidence check
            roi_gradient = gradient_mag[y:y+h, x:x+w]  # 區域梯度
            roi_edges = step_edges[y:y+h, x:x+w]  # 區域邊緣
            
            # 步驟7：基於梯度模式計算信心度
            # Calculate confidence based on gradient patterns
            edge_density = np.sum(roi_edges > 0) / roi_edges.size  # 邊緣密度
            gradient_std = np.std(roi_gradient)  # 梯度標準差
            
            # 假輪廓的特徵：高邊緣密度但低梯度變化
            # False contours have high edge density but low gradient variation
            confidence = edge_density * (1.0 - min(gradient_std, 1.0))
            
            # 步驟8：如果信心度超過閾值，記錄該區域
            if confidence >= self.min_confidence:
                regions.append({
                    'bbox': (x, y, w, h),  # 邊界框
                    'center': (x + w//2, y + h//2),  # 中心點
                    'area': int(area),  # 面積
                    'confidence': confidence  # 信心度
                })
        
        return regions

    def _analyze_gradient_discontinuity(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze gradient discontinuities that indicate false contours.
        
        Args:
            image: Grayscale image array
            
        Returns:
            Gradient discontinuity map
        """
        # Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate second derivatives to detect discontinuities
        grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate Laplacian (second derivative)
        laplacian = grad_xx + grad_yy
        
        # Detect discontinuities as high second derivative values
        discontinuity_map = np.abs(laplacian)
        
        # Normalize
        if discontinuity_map.max() > 0:
            discontinuity_map = discontinuity_map / discontinuity_map.max()
        
        return discontinuity_map
    
    def _detect_step_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect step-like edges characteristic of false contours.
        
        Args:
            image: Grayscale image array
            
        Returns:
            Step edge detection map
        """
        # Use Canny edge detection with multiple thresholds
        edges_low = cv2.Canny((image * 255).astype(np.uint8), 50, 100)
        edges_high = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
        
        # Combine edge maps
        combined_edges = edges_low.astype(np.float32) + edges_high.astype(np.float32)
        combined_edges = np.clip(combined_edges / 255.0, 0, 1)
        
        # Detect step-like patterns using morphological operations
        # False contours often appear as parallel lines or bands
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        
        # Detect horizontal and vertical step patterns
        horizontal_steps = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_horizontal)
        vertical_steps = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Combine step detections
        step_map = np.maximum(horizontal_steps, vertical_steps)
        
        return step_map
    
    def _combine_detection_maps(self, gradient_map: np.ndarray, step_map: np.ndarray) -> np.ndarray:
        """
        Combine gradient discontinuity and step edge maps.
        
        Args:
            gradient_map: Gradient discontinuity map
            step_map: Step edge detection map
            
        Returns:
            Combined detection map
        """
        # Weighted combination of both maps
        combined = 0.6 * gradient_map + 0.4 * step_map
        
        # Apply Gaussian smoothing to reduce noise
        combined = ndimage.gaussian_filter(combined, sigma=1.0)
        
        return combined
    
    def _detect_contour_regions(self, detection_map: np.ndarray, original_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect regions with false contours based on the detection map.
        
        Args:
            detection_map: Combined detection map
            original_image: Original grayscale image
            
        Returns:
            List of detected regions with metadata
        """
        regions = []
        
        # Threshold the detection map
        threshold = np.percentile(detection_map, 85)
        binary_mask = detection_map > threshold
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        for label in range(1, num_labels):
            mask = labels == label
            
            # Calculate region properties
            area = np.sum(mask)
            if area < 50:  # Filter small regions
                continue
            
            # Find bounding box
            coords = np.where(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Calculate confidence based on detection strength and texture consistency
            region_detection = detection_map[mask]
            texture_consistency = self._analyze_texture_consistency(original_image[y_min:y_max+1, x_min:x_max+1])
            
            # Combine detection strength and texture analysis
            detection_confidence = np.mean(region_detection)
            confidence = 0.7 * detection_confidence + 0.3 * texture_consistency
            
            if confidence >= self.min_confidence:
                center_y = (y_min + y_max) // 2
                center_x = (x_min + x_max) // 2
                
                regions.append({
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'center': (center_x, center_y),
                    'area': area,
                    'confidence': confidence
                })
        
        return regions
    
    def _analyze_texture_consistency(self, region: np.ndarray) -> float:
        """
        Analyze texture consistency in a region to identify false contours.
        
        Args:
            region: Image region array
            
        Returns:
            Texture consistency score (higher = more likely false contour)
        """
        if region.size == 0:
            return 0.0
        
        # Calculate local standard deviation
        mean_filter = ndimage.uniform_filter(region, size=5)
        sqr_filter = ndimage.uniform_filter(region**2, size=5)
        local_std = np.sqrt(sqr_filter - mean_filter**2)
        
        # False contours often have regions of very low texture variation
        # followed by sharp transitions
        low_texture_ratio = np.sum(local_std < 0.05) / local_std.size
        
        # Calculate gradient variation
        grad_x = np.gradient(region, axis=1)
        grad_y = np.gradient(region, axis=0)
        gradient_variation = np.std(np.sqrt(grad_x**2 + grad_y**2))
        
        # High low_texture_ratio and low gradient_variation indicate false contours
        consistency_score = low_texture_ratio * (1.0 - min(gradient_variation, 1.0))
        
        return consistency_score
    
    def _calculate_severity(self, confidence: float) -> str:
        """
        Calculate severity level based on confidence score.
        
        Args:
            confidence: Detection confidence
            
        Returns:
            Severity level string
        """
        if confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    def calculate_contour_confidence(self, region: np.ndarray) -> float:
        """
        Calculate confidence score for a specific region.
        
        Args:
            region: Image region array
            
        Returns:
            Confidence score between 0 and 1
        """
        # Analyze the region for false contour characteristics
        texture_score = self._analyze_texture_consistency(region)
        
        # Calculate edge density
        edges = cv2.Canny((region * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine scores
        confidence = 0.6 * texture_score + 0.4 * edge_density
        
        return min(confidence, 1.0)