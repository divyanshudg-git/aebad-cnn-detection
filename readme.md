# 🚀 Aero Engine Blade Anomaly Detection (AeBAD_S, CNN + Transfer Learning)

Anomaly detection on aerospace engine blades using CNN with transfer learning on the AeBAD_S dataset. Handles real-world domain shifts: view, illumination, and background.

---

## 🧠 Features

- Custom PyTorch Dataset class for AeBAD_S
- Multi-defect classification: ablation, breakdown, fracture, groove
- Supports domain shifts
- Ground truth mask overlay for defect visualization

---

## 📁 Dataset

Dataset **not included** (too big for GitHub).  
Mount manually in Kaggle as: /kaggle/input/aebad-s/AeBAD_S

Or link it from [AeBAD paper](https://arxiv.org/abs/2304.02216).

---

## 🧩 Run

All code is in `notebook.ipynb`.  
Upload it, run it on Kaggle. No configs needed.

---

## 📁 Folder Structure

```
├── final-project.ipynb
├── aebad_S.py          # contains AeBAD_SDataset class
├── requirements.txt
└── README.md
```
---

## 🧪 Custom Dataset Class

`AeBAD_SDataset` handles:
- Domain shifts  
- Mask paths for test phase  
- Returns:
  ```python
  [classname, anomaly_type, image_path, mask_path]


## Requirements

torch
torchvision
numpy
matplotlib
Pillow
scikit-learn
tqdm
opencv-python


## Output
Notebook shows sample defect images with masks overlayed.

##
<p align="center"><strong>Made with ❤ by Divyansh Aggarwal and Divyanshu</strong></p>



