"""
Image upload and preprocessing handler for the Image Artifact Detector.
"""
import io
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2
import streamlit as st


class ImageUploadHandler:
    """Handles image upload, validation, and basic information extraction."""
    
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
    
    @staticmethod
    def validate_image(uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded image file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if uploaded_file is None:
            return False, "未選擇檔案"
        
        # Check file size
        if uploaded_file.size > ImageUploadHandler.MAX_FILE_SIZE:
            return False, f"檔案過大，請選擇小於 {ImageUploadHandler.MAX_FILE_SIZE / 1024 / 1024:.0f}MB 的檔案"
        
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in ImageUploadHandler.SUPPORTED_FORMATS:
            return False, f"不支援的檔案格式。支援格式：{', '.join(ImageUploadHandler.SUPPORTED_FORMATS).upper()}"
        
        # Try to open the image
        try:
            image = Image.open(uploaded_file)
            image.verify()  # Verify it's a valid image
            return True, ""
        except Exception as e:
            return False, f"無法讀取影像檔案：{str(e)}"
    
    @staticmethod
    def load_image(uploaded_file) -> Optional[Image.Image]:
        """
        Load and return PIL Image from uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            return image
        except Exception as e:
            st.error(f"載入影像失敗：{str(e)}")
            return None
    
    @staticmethod
    def get_image_info(image: Image.Image, filename: str) -> Dict[str, Any]:
        """
        Extract basic information from the image.
        
        Args:
            image: PIL Image object
            filename: Original filename
            
        Returns:
            Dictionary containing image information
        """
        # Calculate file size (approximate)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        file_size = len(img_byte_arr.getvalue())
        
        return {
            'filename': filename,
            'width': image.width,
            'height': image.height,
            'channels': len(image.getbands()),
            'file_size': file_size,
            'color_mode': image.mode,
            'bit_depth': 8 if image.mode in ['L', 'RGB', 'RGBA'] else 16
        }


class ImagePreprocessor:
    """Handles image preprocessing and format standardization."""
    
    @staticmethod
    def preprocess_image(image: Image.Image, max_size: int = 1024) -> np.ndarray:
        """
        Preprocess image for artifact detection with automatic resizing for performance.
        
        Args:
            image: PIL Image object
            max_size: Maximum dimension for processing (default: 1024)
            
        Returns:
            Preprocessed image as numpy array
        """
        # Resize large images for faster processing
        if max(image.size) > max_size:
            # Calculate new size maintaining aspect ratio
            width, height = image.size
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        return img_array
    
    @staticmethod
    def normalize_image(image_array: np.ndarray) -> np.ndarray:
        """
        Normalize image array to 0-1 range.
        
        Args:
            image_array: Input image array
            
        Returns:
            Normalized image array
        """
        return image_array.astype(np.float32) / 255.0
    
    @staticmethod
    def convert_to_grayscale(image_array: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image_array: Input RGB image array
            
        Returns:
            Grayscale image array
        """
        if len(image_array.shape) == 3:
            # Use OpenCV for better grayscale conversion
            return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        return image_array
    
    @staticmethod
    def resize_image(image_array: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image_array: Input image array
            max_size: Maximum dimension size
            
        Returns:
            Resized image array
        """
        height, width = image_array.shape[:2]
        
        if max(height, width) <= max_size:
            return image_array
        
        # Calculate new dimensions
        if height > width:
            new_height = max_size
            new_width = int(width * max_size / height)
        else:
            new_width = max_size
            new_height = int(height * max_size / width)
        
        # Resize image
        if len(image_array.shape) == 3:
            resized = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized