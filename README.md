#  IU Chest X-ray Report Generator

This repository contains an end-to-end pipeline for generating and refining diagnostic **radiology reports** using **frontal chest X-rays** and **patient history**, based on the **Indiana University Chest X-ray Dataset**.

Built using:
- 🧠 DenseNet-121 for label classification  
- ✍️ Flan-T5-small with LoRA fine-tuning for report generation  
- 📊 BLEU/ROUGE evaluation for clinical accuracy

---

## 🗂️ Project Structure

| Notebook | Purpose |
|----------|---------|
| `IU_Xray_EDA_Preprocessing_Updated.ipynb` | Cleans and merges metadata files; maps MeSH → CheXpert labels |
| `xray-report-generator-optimized.ipynb` | Trains DenseNet, fine-tunes T5 using LoRA, generates & refines reports |
| `IU_Xray_Report_Evaluation_Visualization.ipynb` | Visualizes BLEU/ROUGE metrics, improvement plots, label heatmaps |

---

## 📦 Requirements

**Platform**: [Kaggle Notebooks](https://www.kaggle.com/code)  
**Accelerator**: GPU (T4 × 2 recommended)

### Python Libraries
Install the following if running locally:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets peft accelerate
pip install torchxrayvision scikit-learn matplotlib seaborn
pip install fuzzywuzzy python-Levenshtein
```

---

## 📁 Required Dataset

Download and unzip the dataset:
👉 https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

Ensure the following files/folders are available:
```
IU-Xray/
├── images/
│   └── [DICOM or PNG images]
├── indiana_reports.csv
├── indiana_projections.csv
```

---

## 🚀 How to Run

### Step 1: Preprocessing
Run `IU_Xray_EDA_Preprocessing_Updated.ipynb` to:
- Merge reports and projections
- Filter for **frontal** images
- Map `MeSH` → CheXpert labels
- Output: `merged_preprocessed.csv`

### Step 2: Model Training + Report Generation
Run `xray-report-generator-optimized.ipynb` to:
- Train DenseNet-121 on image-label pairs
- Fine-tune Flan-T5 with LoRA using `indication`, `findings`, and `impression`
- Generate & refine reports
- Output: `final_inference_results.csv`

### Step 3: Evaluation
Run `IU_Xray_Report_Evaluation_Visualization.ipynb` to:
- Plot BLEU/ROUGE score distributions
- Measure improvement after refinement
- Visualize label co-occurrence

---

## 📊 Sample Output
Each image results in:
- ✅ Predicted clinical labels
- 📝 Initial report (based on image)
- 📄 Refined report (using patient history)
- 📈 Evaluation metrics: BLEU / ROUGE

---

## 🧠 Highlights
- Uses pretrained CheXpert DenseNet for fast convergence
- Applies LoRA for lightweight T5 fine-tuning on limited compute
- Handles missing data robustly (`fillna`, prompt fallback)
- Memory-safe inference with `torch.cuda.empty_cache()` + batch flushing

---

## 📌 Future Improvements
- Add early stopping & learning rate scheduling
- Expand to full 7,000+ images (with batch filtering)
- Add multilingual report generation (using mT5 or BART)
- Deploy via Gradio demo for real-time use

---

## 🧑‍💻 Author

Built by Mayur Rattan Jaisinghani
