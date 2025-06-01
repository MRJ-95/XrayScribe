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

Initial Report (Image-only, DenseNet + Flan-T5)
Findings:
There is patchy opacity in the right lower lung field suggestive of consolidation. No pneumothorax or pleural effusion is identified. Cardiac silhouette is within normal limits.

Impression:
Right lower lobe pneumonia. No acute cardiopulmonary abnormality otherwise.

Refined Report (Image + Clinical Context)
Indication:
Cough and fever for 4 days. History of smoking. Comparison to study from 2 days prior.

Comparison:
Compared with prior study dated 2025-05-15.

Findings:
Interval development of dense consolidation in the right lower lobe compared to prior. Mild peribronchial thickening is noted. No pleural effusion or pneumothorax.

Impression:
Progressive right lower lobe pneumonia with new consolidation since prior imaging. Findings consistent with ongoing infectious process.

Evaluation Metrics:
BLEU Score: 0.41

ROUGE-L Score: 0.68

Improvement after context refinement: +0.08 ROUGE, +0.06 BLEU

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


References:
1. Automated Radiology Report Generation using Conditioned Transformers: This work introduces a three-stage pipeline that fine-tunes CheXNet (a DenseNet variant) to predict image tags, computes semantic embeddings from these tags, and conditions a pre-trained GPT-2 model on both visual and semantic features to generate comprehensive reports. 

2. ChestX-Transcribe: This study presents a multimodal transformer model that integrates a Swin Transformer for high-resolution visual feature extraction with DistilGPT for generating clinically relevant reports. The model demonstrates state-of-the-art performance on the IU X-Ray dataset. 
Frontiers

3. Clinical Context-aware Radiology Report Generation from Medical Images using Transformers: This research investigates the use of transformer models for radiology report generation, emphasizing the importance of incorporating clinical context to enhance the coherence and diagnostic value of the generated reports.
