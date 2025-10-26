"""
Image Artifact Detector - Streamlit Application
檢測數位影像中的 Moiré patterns 和 false contours
"""
import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime
import time

# Try to import OpenCV with error handling
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV import failed: {e}")
    st.error("Please check if opencv-python-headless is installed correctly.")
    OPENCV_AVAILABLE = False

# Import our modules with error handling
try:
    from src.image_handler import ImageUploadHandler, ImagePreprocessor
    from src.moire_detector import MoirePatternDetector
    from src.false_contour_detector import FalseContourDetector
    from src.visualization import ArtifactVisualizer
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Module import failed: {e}")
    MODULES_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="影像偽影檢測器",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def process_image(image_array, detect_moire, detect_contours):
    """Process image and detect artifacts (cached for performance)."""
    results = {
        'moire_artifacts': [],
        'contour_artifacts': [],
        'processing_time': 0
    }
    
    start_time = time.time()
    
    # Initialize detectors
    if detect_moire:
        moire_detector = MoirePatternDetector(min_confidence=0.3)
        results['moire_artifacts'] = moire_detector.detect_moire_patterns(image_array)
    
    if detect_contours:
        contour_detector = FalseContourDetector(min_confidence=0.3)
        results['contour_artifacts'] = contour_detector.detect_false_contours(image_array)
    
    results['processing_time'] = time.time() - start_time
    
    return results

def main():
    """Main application entry point."""
    st.title("🔍 影像偽影檢測器")
    st.markdown("檢測數位影像中的 **Moiré patterns（摩爾紋）** 和 **False contours（假輪廓）**")
    
    # Detailed dependency check
    with st.expander("🔧 系統狀態檢查", expanded=not (OPENCV_AVAILABLE and MODULES_AVAILABLE)):
        st.write("**依賴檢查結果：**")
        
        # OpenCV check
        if OPENCV_AVAILABLE:
            try:
                import cv2
                st.success(f"✅ OpenCV: {cv2.__version__}")
            except Exception as e:
                st.error(f"❌ OpenCV 載入錯誤: {e}")
        else:
            st.error("❌ OpenCV: 未安裝或載入失敗")
        
        # Other dependencies
        try:
            import numpy as np
            st.success(f"✅ NumPy: {np.__version__}")
        except ImportError:
            st.error("❌ NumPy: 未安裝")
        
        try:
            from PIL import Image
            st.success(f"✅ Pillow: {Image.__version__}")
        except ImportError:
            st.error("❌ Pillow: 未安裝")
        
        try:
            import scipy
            st.success(f"✅ SciPy: {scipy.__version__}")
        except ImportError:
            st.error("❌ SciPy: 未安裝")
        
        # Module check
        if MODULES_AVAILABLE:
            st.success("✅ 檢測模組: 載入成功")
        else:
            st.error("❌ 檢測模組: 載入失敗")
    
    # Check if all dependencies are available
    if not OPENCV_AVAILABLE or not MODULES_AVAILABLE:
        st.error("❌ 系統依賴未正確安裝，請檢查部署配置")
        st.info("💡 嘗試以下解決方案：")
        st.code("""
        1. 確保 requirements.txt 包含：
           opencv-python-headless==4.8.1.78
           
        2. 如果仍有問題，可能需要 packages.txt：
           libgl1-mesa-glx
           
        3. 重新部署應用程式
        """)
        return
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ 控制面板")
        
        # Detection options
        st.subheader("檢測選項")
        detect_moire = st.checkbox("檢測 Moiré Patterns（摩爾紋）", value=True)
        detect_contours = st.checkbox("檢測 False Contours（假輪廓）", value=True)
        
        # Visualization options
        st.subheader("顯示選項")
        show_original = st.checkbox("顯示原始影像", value=True)
        show_annotated = st.checkbox("顯示標註影像", value=True)
        show_overlay = st.checkbox("顯示疊加效果", value=False)
        
        # Visual legend
        st.subheader("🎨 視覺化說明")
        st.markdown("🔴 **紅色半透明覆蓋**：Moiré Pattern")
        st.markdown("🔵 **藍色半透明覆蓋**：False Contour")
        st.markdown("📍 **圓點 + 數字**：偽影中心和序號")
        st.markdown("👁️ **透明度**：30% 覆蓋，保持原圖可見")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 影像上傳")
        uploaded_file = st.file_uploader(
            "選擇影像檔案",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="支援格式：JPG, PNG, BMP, TIFF（最大 50MB）"
        )
        
        if uploaded_file is not None:
            # Validate uploaded file
            is_valid, error_msg = ImageUploadHandler.validate_image(uploaded_file)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return
            
            # Load and display image
            image = ImageUploadHandler.load_image(uploaded_file)
            if image is None:
                return
            
            # Display basic file info
            image_info = ImageUploadHandler.get_image_info(image, uploaded_file.name)
            st.success(f"✅ 已上傳：{uploaded_file.name}")
            
            # Show image details
            with st.expander("📋 影像資訊"):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**尺寸：** {image_info['width']} × {image_info['height']}")
                    st.write(f"**色彩模式：** {image_info['color_mode']}")
                with col_info2:
                    st.write(f"**檔案大小：** {image_info['file_size'] / 1024 / 1024:.2f} MB")
                    st.write(f"**色彩通道：** {image_info['channels']}")
            
            # Display original image
            if show_original:
                st.image(image, caption="原始影像", width='stretch')
    
    with col2:
        st.subheader("🔍 檢測結果")
        
        if uploaded_file is not None and image is not None:
            if not detect_moire and not detect_contours:
                st.warning("⚠️ 請至少選擇一種檢測類型")
                return
            
            # Process image
            with st.spinner("🔄 正在分析影像..."):
                # Show processing info
                original_size = f"{image.width} × {image.height}"
                
                # Preprocess image (with automatic resizing for performance)
                preprocessor = ImagePreprocessor()
                image_array = preprocessor.preprocess_image(image, max_size=1024)
                
                processed_size = f"{image_array.shape[1]} × {image_array.shape[0]}"
                
                if original_size != processed_size:
                    st.info(f"📏 影像已調整尺寸以提升處理速度：{original_size} → {processed_size}")
                
                # Detect artifacts
                results = process_image(image_array, detect_moire, detect_contours)
            
            # Display results
            moire_artifacts = results['moire_artifacts']
            contour_artifacts = results['contour_artifacts']
            all_artifacts = moire_artifacts + contour_artifacts
            
            # Results summary
            st.subheader("📊 檢測摘要")
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("🔴 摩爾紋", len(moire_artifacts))
            with col_res2:
                st.metric("🔵 假輪廓", len(contour_artifacts))
            with col_res3:
                st.metric("⏱️ 處理時間", f"{results['processing_time']:.2f}s")
            
            # Visualization
            if all_artifacts:
                visualizer = ArtifactVisualizer()
                
                # Create annotated image
                if show_annotated:
                    annotated_image = visualizer.draw_artifact_annotations(image_array, all_artifacts)
                    st.image(annotated_image, caption="檢測結果標註", width='stretch')
                
                # Create overlay visualization
                if show_overlay:
                    overlay_image = visualizer.create_overlay_visualization(image_array, all_artifacts)
                    st.image(overlay_image, caption="疊加顯示", width='stretch')
                
                # Detailed results
                with st.expander("📋 詳細檢測結果"):
                    if moire_artifacts:
                        st.write("**🔴 Moiré Patterns（摩爾紋）：**")
                        for i, artifact in enumerate(moire_artifacts):
                            st.write(f"  {i+1}. 信心度: {artifact.confidence:.3f}, "
                                   f"嚴重程度: {artifact.severity}, "
                                   f"位置: {artifact.center_point}")
                    
                    if contour_artifacts:
                        st.write("**🔵 False Contours（假輪廓）：**")
                        for i, artifact in enumerate(contour_artifacts):
                            st.write(f"  {i+1}. 信心度: {artifact.confidence:.3f}, "
                                   f"嚴重程度: {artifact.severity}, "
                                   f"位置: {artifact.center_point}")
                
                # Download options
                st.subheader("💾 下載結果")
                if st.button("下載標註影像"):
                    # Convert to PIL Image for download
                    annotated_pil = Image.fromarray(annotated_image)
                    
                    # Create download filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"artifact_detection_{timestamp}.png"
                    
                    # Save to bytes
                    import io
                    img_bytes = io.BytesIO()
                    annotated_pil.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    st.download_button(
                        label="📥 下載 PNG 檔案",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/png"
                    )
            else:
                st.success("✅ 未檢測到任何偽影！影像品質良好。")
        
        else:
            st.info("👆 請先上傳影像檔案以開始檢測")


if __name__ == "__main__":
    main()