# 影像偽影檢測器 (Image Artifact Detector)

基於 Streamlit 的數位影像處理應用程式，用於檢測和標示數位影像中的視覺偽影。

## 🎯 功能特色

- 🔍 **自動檢測 Moiré patterns（摩爾紋）** - 使用頻域分析技術
- 📐 **自動檢測 False contours（假輪廓）** - 使用梯度分析和邊緣檢測
- 🎨 **彩色視覺化標註** - 紅色標示摩爾紋，藍色標示假輪廓
- 📊 **詳細檢測報告** - 包含信心度、嚴重程度和位置資訊
- 💾 **下載標註影像** - PNG 格式輸出
- ⚡ **即時處理** - 快速分析和結果顯示

## 🚀 安裝與執行

### 本地執行
1. **安裝依賴套件：**
```bash
pip install -r requirements.txt
```

2. **執行應用程式：**
```bash
streamlit run app.py
```

3. **測試功能：**
```bash
python test_app.py
```

### Streamlit Cloud 部署
1. **確保包含以下文件：**
   - `requirements.txt` - Python 依賴
   - `packages.txt` - 系統依賴
   - `.streamlit/config.toml` - Streamlit 配置

2. **重要依賴：**
   - 使用 `opencv-python-headless` 而非 `opencv-python`
   - 包含必要的系統庫依賴

3. **部署後檢查：**
   - 確認所有依賴正確安裝
   - 檢查錯誤日誌以排除問題

## 🎨 視覺化標記

- 🔴 **紅色邊框 + 數字**：Moiré Pattern（摩爾紋）
- 🔵 **藍色邊框 + 數字**：False Contour（假輪廓）
- 📍 **中心點標記**：每個偽影的精確位置
- 🔢 **數字編號**：同類型偽影的序號（1, 2, 3...）

## 📁 專案結構

```
├── app.py                          # 主要 Streamlit 應用程式
├── test_app.py                     # 功能測試腳本
├── requirements.txt                # Python 依賴套件
├── src/                           # 核心程式碼
│   ├── __init__.py
│   ├── models.py                  # 資料模型定義
│   ├── image_handler.py           # 影像上傳和預處理
│   ├── moire_detector.py          # Moiré pattern 檢測器
│   ├── false_contour_detector.py  # False contour 檢測器
│   └── visualization.py           # 視覺化和標註功能
├── tests/                         # 單元測試
└── assets/                        # 測試影像和資源
```

## 🔧 技術規格

### 支援格式
- **影像格式：** JPG, JPEG, PNG, BMP, TIFF
- **檔案大小：** 最大 50MB
- **色彩模式：** RGB, RGBA, 灰階

### 檢測演算法
- **Moiré Pattern 檢測：** FFT 頻域分析 + 頻率峰值檢測
- **False Contour 檢測：** Sobel/Canny 邊緣檢測 + 梯度不連續性分析

### 效能特色
- **自動影像尺寸調整**：大圖自動縮放至 1024px 以提升速度
- **優化演算法**：快速 FFT 和梯度分析
- **處理時間**：通常 < 1 秒（1024px 以下影像）
- **記憶體優化**：避免大圖記憶體溢出

## 📖 使用說明

1. **上傳影像：** 點擊「選擇影像檔案」按鈕上傳要分析的影像
2. **選擇檢測類型：** 在側邊欄選擇要檢測的偽影類型
3. **查看結果：** 系統會自動分析並顯示檢測結果
4. **下載結果：** 點擊「下載標註影像」保存分析結果

## 🔬 檢測原理

### Moiré Pattern 檢測
- 使用 FFT 將影像轉換到頻域
- 檢測異常的頻率峰值
- 將頻域特徵映射回空間域
- 計算檢測信心度和嚴重程度

### False Contour 檢測
- 使用 Sobel 和 Canny 演算法檢測邊緣
- 分析梯度不連續性
- 檢測階梯狀邊緣模式
- 評估紋理一致性

## 🧪 測試範例

執行 `python test_app.py` 會創建一個包含潛在偽影的測試影像，並展示檢測功能。測試結果會保存為 `test_detection_result.png`。

## 📋 系統需求

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- Streamlit 1.28+
- Pillow 10.0+
- SciPy 1.11+