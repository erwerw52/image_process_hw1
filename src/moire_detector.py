"""
摩爾紋檢測模組 - 使用頻域分析技術檢測影像中的摩爾紋偽影
Moiré Pattern detection module using frequency domain analysis.

摩爾紋是當兩個或多個規律圖案重疊時產生的干涉圖案，
通常出現在數位化過程中採樣頻率不足時。
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, fftshift
from typing import List, Dict, Any, Tuple
from src.models import ArtifactResult


class MoirePatternDetector:
    """
    摩爾紋檢測器 - 使用頻域分析檢測影像中的摩爾紋
    Detects Moiré patterns in images using frequency domain analysis.
    
    檢測原理：
    1. 將影像轉換到頻域（FFT）
    2. 分析頻率成分，尋找異常的週期性模式
    3. 將頻域特徵映射回空間域
    4. 識別摩爾紋區域並計算信心度
    """
    
    def __init__(self, min_confidence: float = 0.3):
        """
        初始化摩爾紋檢測器
        Initialize the Moiré pattern detector.
        
        Args:
            min_confidence: 最小信心度閾值，低於此值的檢測結果將被過濾
                          Minimum confidence threshold for detection
        """
        self.min_confidence = min_confidence
    
    def detect_moire_patterns(self, image: np.ndarray) -> List[ArtifactResult]:
        """
        檢測輸入影像中的摩爾紋
        Detect Moiré patterns in the input image.
        
        Args:
            image: 輸入影像陣列（灰階或彩色）
                  Input image array (grayscale or RGB)
            
        Returns:
            檢測到的摩爾紋偽影列表
            List of detected Moiré pattern artifacts
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
        
        # 步驟4：使用快速檢測演算法
        # Use simplified detection for speed
        moire_regions = self._fast_moire_detection(gray_image)
        
        # 步驟5：如果影像被縮放，將座標縮放回原始尺寸
        # Scale back coordinates if image was resized
        if scale_factor != 1.0:
            for region in moire_regions:
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
        for region in moire_regions:
            artifact = ArtifactResult(
                artifact_type="moire",
                bounding_box=region['bbox'],
                confidence=region['confidence'],
                severity=self._calculate_severity(region['confidence']),
                center_point=region['center'],
                area=region['area']
            )
            artifacts.append(artifact)
        
        return artifacts
    
    def _extract_frequency_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract frequency domain features using FFT.
        
        Args:
            image: Grayscale image array
            
        Returns:
            Frequency feature map
        """
        # Apply FFT
        f_transform = fft2(image)
        f_shift = fftshift(f_transform)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(f_shift)
        
        # Apply logarithmic scaling
        log_spectrum = np.log(magnitude_spectrum + 1)
        
        # Create frequency coordinate grids
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        
        # Create radial frequency map
        y, x = np.ogrid[:h, :w]
        radial_freq = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Detect periodic patterns in frequency domain
        # Look for peaks that indicate regular patterns (potential Moiré)
        frequency_peaks = self._detect_frequency_peaks(log_spectrum, radial_freq)
        
        return frequency_peaks
    
    def _detect_frequency_peaks(self, spectrum: np.ndarray, radial_freq: np.ndarray) -> np.ndarray:
        """
        Detect frequency peaks that indicate Moiré patterns.
        
        Args:
            spectrum: Magnitude spectrum
            radial_freq: Radial frequency map
            
        Returns:
            Peak detection map
        """
        # Smooth the spectrum
        smoothed_spectrum = ndimage.gaussian_filter(spectrum, sigma=2)
        
        # Find local maxima
        local_maxima = ndimage.maximum_filter(smoothed_spectrum, size=5) == smoothed_spectrum
        
        # Filter peaks by intensity and frequency range
        # Moiré patterns typically appear in mid-frequency ranges
        min_freq, max_freq = 10, min(spectrum.shape) // 4
        freq_mask = (radial_freq >= min_freq) & (radial_freq <= max_freq)
        
        # Threshold for significant peaks
        intensity_threshold = np.percentile(smoothed_spectrum, 95)
        intensity_mask = smoothed_spectrum > intensity_threshold
        
        # Combine masks
        peak_mask = local_maxima & freq_mask & intensity_mask
        
        # Create peak strength map
        peak_map = np.zeros_like(spectrum)
        peak_map[peak_mask] = smoothed_spectrum[peak_mask]
        
        return peak_map
    
    def _detect_moire_regions(self, frequency_map: np.ndarray, original_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect spatial regions with Moiré patterns based on frequency analysis.
        
        Args:
            frequency_map: Frequency peak map
            original_image: Original grayscale image
            
        Returns:
            List of detected regions with metadata
        """
        regions = []
        
        # Convert frequency peaks back to spatial domain
        # Use inverse FFT to localize the patterns
        spatial_response = self._frequency_to_spatial(frequency_map, original_image.shape)
        
        # Threshold the response
        threshold = np.percentile(spatial_response, 90)
        binary_mask = spatial_response > threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
        
        for label in range(1, num_labels):
            mask = labels == label
            
            # Calculate region properties
            area = np.sum(mask)
            if area < 100:  # Filter small regions
                continue
            
            # Find bounding box
            coords = np.where(mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Calculate confidence based on response strength
            region_response = spatial_response[mask]
            confidence = np.mean(region_response) / np.max(spatial_response)
            
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
    
    def _fast_moire_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        快速摩爾紋檢測 - 使用滑動窗口和簡化頻域分析
        Fast Moiré pattern detection using simplified frequency analysis.
        
        檢測策略：
        1. 使用小窗口滑動掃描整個影像
        2. 對每個窗口進行快速 FFT 分析
        3. 檢測中頻段的能量集中度
        4. 識別具有週期性模式的區域
        
        Args:
            image: 灰階影像陣列
                  Grayscale image array
            
        Returns:
            檢測到的區域列表
            List of detected regions
        """
        regions = []
        
        # 步驟1：確定滑動窗口大小（為了速度使用較小的窗口）
        # Use smaller FFT window for speed
        h, w = image.shape
        window_size = min(128, min(h, w) // 2)  # 最大128像素，或影像尺寸的一半
        
        if window_size < 32:
            return regions  # 影像太小，無法進行有效檢測
        
        # 步驟2：設定滑動步長（窗口大小的一半，確保重疊檢測）
        # Slide window across image
        step_size = window_size // 2
        
        # 步驟3：滑動窗口掃描整個影像
        for y in range(0, h - window_size, step_size):
            for x in range(0, w - window_size, step_size):
                # 提取當前窗口
                # Extract window
                window = image[y:y+window_size, x:x+window_size]
                
                # 步驟4：對窗口進行快速摩爾紋檢測
                # Quick frequency analysis
                confidence = self._quick_moire_check(window)
                
                # 步驟5：如果信心度超過閾值，記錄該區域
                if confidence >= self.min_confidence:
                    regions.append({
                        'bbox': (x, y, window_size, window_size),  # 邊界框
                        'center': (x + window_size//2, y + window_size//2),  # 中心點
                        'area': window_size * window_size,  # 面積
                        'confidence': confidence  # 信心度
                    })
        
        return regions
    
    def _quick_moire_check(self, window: np.ndarray) -> float:
        """
        快速摩爾紋檢測 - 分析小窗口中的頻域特徵
        Quick check for Moiré patterns in a small window.
        
        檢測原理：
        摩爾紋通常在中頻段產生強烈的週期性信號，
        通過分析頻域能量分佈可以識別這些模式。
        
        Args:
            window: 小影像窗口
                   Small image window
            
        Returns:
            信心度分數（0-1）
            Confidence score
        """
        # 步驟1：對小窗口進行快速 FFT 變換
        # Simple FFT on small window
        f_transform = np.fft.fft2(window)
        magnitude = np.abs(f_transform)  # 取頻譜幅度
        
        # 步驟2：移除直流分量（DC component）
        # Look for strong periodic components
        # Remove DC component
        magnitude[0, 0] = 0  # 直流分量不包含摩爾紋資訊
        
        # 步驟3：計算不同頻段的能量分佈
        # Calculate energy in different frequency bands
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2  # 頻域中心
        
        # 步驟4：建立徑向頻率映射
        # Create frequency rings
        y, x = np.ogrid[:h, :w]
        radial_freq = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # 步驟5：分析中頻段能量（摩爾紋通常出現在此頻段）
        # Mid-frequency energy (where Moiré typically appears)
        min_freq = 5  # 最小頻率（避免低頻雜訊）
        max_freq = min(h, w) // 4  # 最大頻率（避免高頻雜訊）
        mid_freq_mask = (radial_freq >= min_freq) & (radial_freq <= max_freq)
        mid_freq_energy = np.sum(magnitude[mid_freq_mask])  # 中頻段總能量
        
        # 步驟6：計算總能量
        # Total energy
        total_energy = np.sum(magnitude)
        
        if total_energy == 0:
            return 0.0  # 避免除零錯誤
        
        # 步驟7：基於中頻段能量集中度計算信心度
        # Confidence based on mid-frequency concentration
        # 摩爾紋的特徵是中頻段能量集中
        energy_ratio = mid_freq_energy / total_energy
        confidence = min(energy_ratio * 10, 1.0)  # 放大比例並限制在 0-1 範圍
        
        return confidence

    def _frequency_to_spatial(self, frequency_map: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert frequency domain features back to spatial domain.
        
        Args:
            frequency_map: Frequency domain feature map
            image_shape: Target image shape
            
        Returns:
            Spatial response map
        """
        # Create a filter based on detected peaks
        h, w = frequency_map.shape
        filter_kernel = np.zeros((h, w), dtype=np.complex128)
        
        # Use frequency peaks as filter weights
        peak_locations = frequency_map > 0
        filter_kernel[peak_locations] = frequency_map[peak_locations]
        
        # Apply inverse FFT to get spatial response
        spatial_response = np.abs(np.fft.ifft2(np.fft.ifftshift(filter_kernel)))
        
        # Resize to match original image if needed
        if spatial_response.shape != image_shape:
            spatial_response = cv2.resize(spatial_response, (image_shape[1], image_shape[0]))
        
        return spatial_response
    
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
    
    def calculate_moire_confidence(self, region: np.ndarray) -> float:
        """
        Calculate confidence score for a specific region.
        
        Args:
            region: Image region array
            
        Returns:
            Confidence score between 0 and 1
        """
        # Analyze frequency content of the region
        f_transform = fft2(region)
        magnitude = np.abs(f_transform)
        
        # Look for periodic patterns
        # High confidence if strong periodic components are found
        sorted_magnitudes = np.sort(magnitude.flatten())[::-1]
        
        # Calculate ratio of peak magnitudes to mean
        peak_ratio = sorted_magnitudes[:10].mean() / sorted_magnitudes.mean()
        
        # Normalize to 0-1 range
        confidence = min(peak_ratio / 100.0, 1.0)
        
        return confidence