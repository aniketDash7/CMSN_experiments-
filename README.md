# Report: Cloud Removal using CMSN - Earth Engine and PyTorch

## Overview
This report summarizes the implementation and experimentation process described in the notebook `CRearthengine.ipynb`, which focuses on reconstructing cloud-free satellite imagery using the Coarse-to-Fine Multi-temporal Satellite Network (CMSN). The dataset is built using Google Earth Engine (GEE) and processed through a deep learning pipeline using PyTorch.

---

## How to Run the Notebook

### Requirements
- Google Colab or local setup with GPU support
- Installed Python packages:
  - `torch`, `torchvision`, `numpy`, `matplotlib`, `rasterio`, `scikit-image`, `earthengine-api`, `geemap`

### Steps
1. **Authenticate and initialize Earth Engine** using your Google account.
2. **Run the data extraction pipeline** using GEE to extract patches from Landsat 5 and Landsat 8 images over regions like Australia and Wuhan.
3. **Export GeoTIFFs to Google Drive**, then manually download and place them in a local `CMSN_Dataset` folder.
4. **Initialize Dataset and DataLoader** using `CMSNDataset`.
5. **Define and initialize the CMSN model** with ResFFTConv layers and MFAM fusion.
6. **Train the model** using `train_cmsn()` with specified hyperparameters (epochs, learning rate, scheduler).
7. **Visualize predictions** and compute NDVI, PSNR, and SSIM for evaluation.

---

## Notebook Structure Summary

### 1. Dataset Preparation
- **GEE-based Extraction**: Cloudy patches are extracted with controlled cloud cover using QA_PIXEL masks.
- **Patch Generation**: Fixed-size image patches are clipped using GEE's buffer and bounds method.

### 2. Model Architecture
- **ResFFTConv**: Performs 2D FFT followed by convolution and residual skip connection.
- **MFAM**: Multi-temporal feature attention module fuses features from T1, T2, T3.
- **CMSN**: Coarse output is decoded and refined through an additional convolution layer with tanh activation.

### 3. Loss Function
- **CMSNLoss** combines:
  - Global-local L1 loss (cloud and clear regions)
  - Frequency reconstruction loss (via FFT)
  - Refinement loss (L1 between final output and ground truth)

### 4. Training Pipeline
- Learning rate: `5e-3`, StepLR scheduler (step size: 10, gamma: 0.8)
- Batch size: 4, 200 epochs
- Batch normalization and tanh activation added for stability

### 5. Evaluation
- NDVI computed using Red and NIR bands from output vs ground truth
- PSNR and SSIM calculated for quantitative evaluation of prediction quality
- Natural color visualization created from RGB (B4, B3, B2 for Landsat 8)

---

## Challenges and Solutions

| Challenge | Solution |
|----------|----------|
| **NaN losses during training** | Replaced NaNs with zeros during preprocessing using `np.nan_to_num`. Also added NaN check during training to skip problematic batches. |
| **Low contrast predictions** | Added `torch.tanh` in the final output layer and applied normalization in FFT processing to stabilize gradients. |
| **Unrealistic NDVI values** | Added range clipping in `calculate_ndvi()` to limit NDVI values to [-1, 1]. Normalized Red/NIR bands if above range. |
| **Colorless/gray visualizations** | Adjusted visualization code to map bands to natural color (B4, B3, B2), using normalization and clipping. |
| **Gradient vanishing** | Added `BatchNorm2d` after each convolution to stabilize learning and avoid vanishing gradients. |
| **Oscillating loss** | Tuned learning rate schedule and added `StepLR` scheduler to reduce learning rate gradually and avoid overfitting. |
| **Cloud mask not affecting results** | Refined GEE cloud masking using all relevant QA_PIXEL bits and used a probabilistic mask scaled to desired cloud fraction. |

---

## Conclusion
The CMSN model effectively reconstructs cloud-free satellite images by leveraging frequency domain cues and temporal fusion. With improved preprocessing, careful loss design, and network stabilization using batch norm and tanh activation, the model achieves visually and quantitatively better performance.

Next steps include adding more diverse geographic regions, experimenting with attention mechanisms, and possibly integrating domain-specific indices like NDWI or EVI into training or evaluation.

