# Plexor AI Engineer Test - Object Detection using YOLO

This is my submission for the Plexor Junior AI Engineer/Intern Practical Implementation Test

## Overview

Detects suspicious behavior in retail environments:
- **Person**: Individuals in frame
- **Item**: Products being handled/concealed

**Performance**: 92.1% mAP@50, 95% precision, 86% recall

---

## 📁 Repository Structure
```
plexor_ai_engineer_code/
 notebooks/                        # Complete training pipeline
├── dataset/                       # Sample labeled data
├── weights/best.pt                # Trained model weights (22.5MB)
├── annotated_outputs/             # Detection results
└── REPORT.pdf                     # Summary report
```

---

### Environment Setup

**Platform**: Google Colab (NVIDIA T4 GPU)
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install ultralytics
```

### Full Pipeline

1. **Open and run notebooks in Colab in the following order**:
   `notebooks/extract_frames.ipynb`
   `notebooks/plexor_training.ipynb`
   `notebooks/plexor_inference.ipynb`
   
2. **Organize data in Google Drive**:
   **This is important to be accurate in order to run the notebooks in colab**
```
   plexor_ai_engineer/
   ├── videos/              # Place source videos here
   ├── frames/              # Extracted frames
      ├── plexor_fridge_theft/ #Extracted frames for fridge video
      ├── plexor_shelf_theft/  #extracted frames for shelf video
   ├── labels_generated/  # Exported labels from Makesense.ai
   ├── dataset/             # Organized train/val split
      ├── images/
          ├── train/
          ├── val/
      ├── labels/
          ├── train/
          ├── val/
   ├── versions/            # Saves model versions
   ├── outputs/             # Saves annotated videos
   └── weights/             # Trained model output

```
4. **Results saved** to `weights/best.pt` and `annotated_outputs/`

---

## 📊 Dataset

- **Source**: 2 CCTV videos (from Plexor)
- **Frames extracted**: 348 total
  - Fridge area: 246 frames (interval: 10)
  - Shelf area: 102 frames (interval: 2)
- **Classes**: `person`, `item`
- **Split**: 80% train (278), 20% val (70)
- **Labeling tool**: [Makesense.ai](https://makesense.ai)

---

## 🎓 Training Details

**Model**: YOLOv8s (pre-trained on COCO)

**Hardware**: Google Colab T4 GPU (16GB VRAM)

**Optimized Hyperparameters**:
```python
epochs=100
batch=8
imgsz=640
device=cuda
```


## 📈 Results

| Metric | Value |
|--------|-------|
| mAP@50 | 92.1% |
| Precision | 95.0% |
| Recall | 86.4% |
| Inference Speed | 10.55 ms/frame (~95 FPS) |
| Model Size | 22.5 MB |

**Key Achievement**: Improvement in mAP@50 jumped from 52% to 92% through optimization.

---

## 🔍 Inference

**Using trained model**:
```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')
results = model.predict(
    source='path/to/video.mp4',
    save=True,
    conf=0.25,
    stream=True
)
```

Annotated outputs available in `annotated_outputs/`

---

## 🛠️ Key Challenges

1. **Limited dataset** (348 frames, single subject)
2. **Class imbalance** (person: ~348 instances, item: ~80)
3. **Compute constraints** (Colab timeout on long videos)

**Solutions**: Batch size tuning, streaming inference, aggressive augmentation

---

## 📝 Documentation

Full technical report: [`docs/REPORT.pdf`](docs/REPORT.pdf)

---


## 📦 Dependencies
### Most dependencies are already installed in Google Colab
```
ultralytics
```


