# Cloud Removal With CMSN

This report documents the implementation and experimentation of the CMSN (Coarse-to-refined multi-temporal synchronous cloud removal network) model as described in the referenced paper. The primary goal is to reconstruct cloud-free remote sensing images from multi-temporal, cloud-contaminated Landsat imagery using frequency and spatial domain learning.

---

## Setup Instructions (Steps to Run Notebook)

1. **Environment Setup**:
   - Google Colab or local Python environment with GPU recommended
   - Required libraries:
     ```bash
     pip install torch torchvision rasterio matplotlib earthengine-api scikit-image
     ```

2. **Google Earth Engine Authentication**:
   - Authenticate and initialize `ee` (Earth Engine Python API) in Colab

3. **Run Preprocessing Blocks**:
   - Define `regions`, date ranges, and import Landsat imagery
   - Apply QA-based cloud masking and create clipped patches
   - Export cloud-controlled patches to Google Drive

4. **Download Dataset**:
   - Use exported `.tif` files from Google Drive and organize them locally
   - Expected naming convention: `Region_T{1,2,3}_Patch{i}.tif`

5. **Model Training**:
   - Define CMSN architecture
   - Train using multi-temporal inputs with global-local, frequency, and refinement losses
   - Plot loss trends over 200 epochs

6. **Evaluation**:
   - Visualize model predictions
   - Calculate NDVI, PSNR, and SSIM for prediction quality

---

##  Dataset Preparation

- **Regions Used**:
  - `New South Wales, Australia`: 093/084
  - `Wuhan, China`: 123/039

- **Imagery Sources**:
  - Landsat 5 and 8 (TOA collections)
  - Spectral Bands: B2 (Blue), B3 (Green), B4 (Red), B5 (NIR)

- **Cloud Masking**:
  - Utilized `QA_PIXEL` band (bitwise flags: cloud, shadow, cirrus)
  - Controlled cloud levels: 25%, 15%, and 10% for T1, T2, and T3 respectively

- **Patch Extraction**:
  - Fixed-size 256x256 pixel patches at 30m resolution
  - Exported using `ee.batch.Export.image.toDrive()`

---

##  CMSN Model Architecture

- **ResFFTConv**:
  - Applies 2D FFT, followed by two convolutions + skip connection
  - Normalized frequency features and batch normalization added to stabilize gradients

- **MFAM (Multi-temporal Feature Aggregation Module)**:
  - Concatenates multi-temporal features and applies spatial attention via conv layers

- **CMSN Decoder & Refinement**:
  - Produces coarse reconstruction first
  - Applies one more conv layer with `tanh` to produce refined output

---

##  Loss Function (CMSNLoss)

- **Multi-temporal Global-Local Loss**: Preserves clear regions and reconstructs cloudy ones
- **Frequency Reconstruction Loss**: Uses 2D FFT to ensure frequency alignment
- **Refinement Loss**: Final L1 loss between refined output and ground truth
- Weighted combo: `loss = mtgl + delta * mtfr + xi * refine`

---

## Evaluation Metrics

- **NDVI (Normalized Difference Vegetation Index)**:
  ```python
  NDVI = (NIR - RED) / (NIR + RED + 1e-8)
  ```
  - Compared predicted NDVI vs ground truth

- **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**:
  - Calculated between predicted and reference images using `skimage.metrics`

- **Visualization**:
  - RGB composite (B4, B3, B2 → Red, Green, Blue)
  - `t1`, `t2`, `t3`, `coarse`, `refined`, and `difference` plots

---

##  Challenges and Solutions

### 1. **NaN Values in Training**
- **Issue**: NaN loss during training due to image patches containing invalid pixels
- **Solution**: Replaced NaNs with 0 during preprocessing using `np.nan_to_num`

### 2. **Low Gradient Flow**
- **Issue**: Small gradient magnitudes (< 1e-3) hindered learning
- **Solution**: Introduced `BatchNorm2d` layers in `ResFFTConv` to stabilize training

### 3. **Color Distortion in Outputs**
- **Issue**: Output predictions appeared grey or washed-out
- **Solution**: Used natural color composite (bands B4, B3, B2) for visualization

### 4. **Incorrect NDVI Ranges**
- **Issue**: NDVI values > 1 or NaN due to unnormalized bands
- **Solution**: Normalized band values and clamped NDVI to [-1, 1]

### 5. **Loss Oscillation**
- **Issue**: Loss plateaued and oscillated after many epochs
- **Solution**: Tuned loss weights (`lambda_gl`, `delta`, `xi`) and learning rate

### 6. **FFT Input Shape Mismatch**
- **Issue**: Concatenated real and imag parts of FFT doubled input channels
- **Solution**: Used `abs()` of FFT instead of concatenation for shape compatibility

---

##  Suggestions for Future Work

- Add dropout to avoid overfitting
- Experiment with different optimizers (e.g., AdamW, Ranger)
- Increase dataset size by including more geographic regions (e.g., Tongchuan)
- Use attention modules or Swin Transformers for global context
- Evaluate on actual test data with real cloud-free targets

---

**Author**: Aniket  
**Project**: Cloud-aware Multi-temporal Satellite Network (CMSN)  
**Date**: April 2025

