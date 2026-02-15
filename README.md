# Derma-Fusion-Net (v9)

**Author:** Om Patil  
**GitHub:** https://github.com/Om-Patil-04

A complete end-to-end deep learning pipeline for melanoma classification and thickness prediction using dermoscopic images.

---

## 📋 Project Overview

Derma-Fusion-Net is a multi-task deep learning framework that performs simultaneous binary classification (Benign vs Malignant) and thickness regression (Breslow depth in mm) on dermoscopic images. The model leverages a DINO Vision Transformer backbone with a dual-head architecture for joint learning, enabling clinically relevant predictions in dermatology.

**Key Features:**
- Binary Classification (Benign vs Malignant melanoma)
- Thickness Regression (Breslow depth prediction in mm)
- DINO-ViT backbone with fine-tuning capabilities
- Multi-task learning with combined loss optimization
- Complete inference pipeline with single image prediction

---

## 🗂️ Project Structure
```
DERMA-FUSION-NET/
├── notebooks/
│   ├── 01_backbone_model.ipynb      # Classification model training
│   ├── 02_dual_task_model.ipynb     # Multi-task model training
│   └── 03_model_evaluation.ipynb    # Inference and evaluation
├── backbone_model.pth               # Saved classification model
├── dual_model.pth                   # Saved multi-task model
├── requirements.txt
├── README.md
└── data/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── train/images/
    ├── val/images/
    └── test/images/
```

---

## 📓 Notebook Workflow

### 1. **01_backbone_model.ipynb** — Classification Training
Trains a DINO Vision Transformer for binary melanoma classification.

**What it does:**
- Fine-tunes DINO-ViT backbone on dermoscopic images
- Applies image augmentation for robustness
- Uses CrossEntropyLoss for binary classification
- Implements early stopping and learning rate scheduling
- Saves best model checkpoint

**Output:**
```
backbone_model.pth
```

**Expected metrics:**
- Accuracy, Precision, Recall, F1-score, ROC-AUC

---

### 2. **02_dual_task_model.ipynb** — Multi-Task Training
Builds and trains a dual-head architecture combining classification and regression.

**What it does:**
- Constructs multi-task model (classification + regression heads)
- Normalizes thickness values for stable training
- Combines CrossEntropyLoss (classification) and MSE Loss (regression)
- Applies weighted loss balancing between tasks
- Trains dual-head architecture end-to-end

**Output:**
```
dual_model.pth
training_metrics/
├── Classification + Thickness
├── Testing
└── plots/
```

**Expected metrics:**
- Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Regression: MAE, RMSE, MAPE

---

### 3. **03_model_evaluation.ipynb** — Inference & Evaluation
Loads trained model and performs predictions on single images and datasets.

**What it does:**
- Loads the trained dual-task model
- Performs single image inference
- Predicts both class (benign/malignant) and thickness
- Generates evaluation metrics and visualizations
- Exports results to CSV

**Example Output:**
```
Predicted Class: malignant
Predicted Thickness: 1.0285 mm
Confidence: 0.92
```

---

## 📊 Model Architecture

### Backbone: DINO Vision Transformer
- Pre-trained DINO-ViT for feature extraction
- Fine-tuned on dermoscopic image dataset
- Dimensions: [batch_size, 768] feature vectors

### Dual-Head Design
```
Input Image
    ↓
DINO-ViT Backbone
    ↓
Shared Features (768-dim)
    ├─→ Classification Head → Benign/Malignant
    └─→ Regression Head → Thickness (mm)
```

---

## 📈 Metrics & Performance

### Classification Metrics
| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve |

### Regression Metrics
| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (mm) |
| **RMSE** | Root Mean Squared Error (mm) |
| **MAPE** | Mean Absolute Percentage Error (%) |

---

## 📂 Data Organization

Your dataset should be organized as follows:
```
data/
├── train.csv          # Training metadata
├── val.csv            # Validation metadata
├── test.csv           # Testing metadata
├── train/images/      # Training images
├── val/images/        # Validation images
└── test/images/       # Testing images
```

### CSV Format Requirements

Each CSV file must contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| **image** | str | Image filename or path |
| **label** | int | 0 = Benign, 1 = Malignant |
| **thickness** | float | Breslow depth in millimeters |

**Example:**
```csv
image,label,thickness
IMG_001.jpg,0,0.45
IMG_002.jpg,1,1.23
IMG_003.jpg,0,0.67
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- pip or conda

### Step 1: Clone Repository
```bash
git clone https://github.com/Om-Patil-04/Derma-Fusion-Net.git
cd Derma-Fusion-Net
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Or using conda
conda create -n derma-fusion python=3.9
conda activate derma-fusion
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

---

## ▶️ How to Run

### Training Workflow

**Step 1: Train Classification Model**
```
Open and run: notebooks/01_backbone_model.ipynb
```
This trains the DINO-ViT backbone for binary classification and saves `backbone_model.pth`.

**Step 2: Train Multi-Task Model**
```
Open and run: notebooks/02_dual_task_model.ipynb
```
This trains the dual-head architecture combining classification and regression, saving `dual_model.pth`.

**Step 3: Evaluate & Predict**
```
Open and run: notebooks/03_model_evaluation.ipynb
```
This loads the trained model and performs inference on test images.

### Quick Inference (Pseudocode)
```python
import torch
from model import DermaFusionNet

# Load model
model = DermaFusionNet()
model.load_state_dict(torch.load('dual_model.pth'))
model.eval()

# Predict
with torch.no_grad():
    class_pred, thickness_pred = model(input_image)
    
print(f"Class: {'Malignant' if class_pred > 0.5 else 'Benign'}")
print(f"Thickness: {thickness_pred:.2f} mm")
```

---

## 🔧 Configuration & Hyperparameters

Key training parameters (adjustable in notebooks):

| Parameter | Default | Notes |
|-----------|---------|-------|
| Batch Size | 32 | Adjust based on GPU memory |
| Learning Rate | 1e-4 | For backbone fine-tuning |
| Epochs | 50 | Early stopping may trigger earlier |
| Loss Weights | α=1.0, β=0.5 | Weights for classification and regression |
| Optimizer | AdamW | Recommended for ViT |
| Image Size | 224×224 | Standard for Vision Transformers |

---

## 📝 Output Files

After successful training, the following files are generated:
```
backbone_model.pth              # Classification model weights
dual_model.pth                  # Multi-task model weights
training_metrics/
├── classification_metrics.csv  # Per-epoch classification metrics
├── regression_metrics.csv      # Per-epoch regression metrics
├── confusion_matrix.png        # Confusion matrix visualization
├── roc_curve.png              # ROC curve plot
└── thickness_distribution.png # Predicted vs. actual thickness
```

---

## 🔍 Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size in notebook configuration or use gradient accumulation.

### Issue: Low Classification Accuracy
**Solution:** 
- Check data balance (benign vs. malignant)
- Verify image preprocessing and normalization
- Increase training epochs or adjust learning rate

### Issue: High Regression Error (MAE/RMSE)
**Solution:**
- Verify thickness values are normalized correctly
- Check for outliers in thickness distribution
- Consider adjusting loss weights (β parameter)

### Issue: Model Doesn't Load
**Solution:** Ensure the model architecture matches the saved weights exactly. Verify DINO-ViT version compatibility.

---

## 📚 References & Dependencies

### Key Libraries
- **PyTorch** — Deep learning framework
- **torchvision** — Vision model utilities
- **timm** — Vision Transformer implementations (DINO)
- **scikit-learn** — Metrics and evaluation
- **pandas** — Data handling
- **matplotlib/seaborn** — Visualization

### Citation
If you use this project in research, please cite:
```bibtex
@software{derma_fusion_net_2024,
  title={Derma-Fusion-Net: Multi-task Deep Learning for Melanoma Classification and Thickness Prediction},
  author={Patil, Om},
  url={https://github.com/Om-Patil-04/Derma-Fusion-Net},
  year={2024}
}
```

---

## 📄 License

MIT License — See LICENSE file for full details.
```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, and distribute copies of the Software.
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 💬 Contact & Support

- **Author:** Om Patil
- **GitHub:** https://github.com/Om-Patil-04
- **Issues:** Please report bugs via GitHub Issues

For questions or discussions about the project, feel free to reach out!

---

**Last Updated:** February 2024  
**Version:** 9.0

