
# DISR_ViTs — Document Image Shadow Removal with Vision Transformers

A research repository for **document image shadow removal** using **Vision Transformers (ViTs)** with **background guidance**.  
This project follows the paper draft included in this repo and integrates two complementary ideas:
- **Color-aware Background Extraction (CBENet)** to estimate a clean background prior.
- **Background-Guided Shadow Network (BGShadowNet)** to focus restoration on shadowed regions.
- A **ViT-based encoder–decoder** to capture long-range dependencies crucial for text regions.

> If you’re here just to run inference on your own images, jump to **[Quickstart → Inference](#inference)**.

---

## Table of Contents
- [News](#news)
- [Highlights](#highlights)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results & Models](#results--models)
- [Reproducibility Notes](#reproducibility-notes)
- [Paper](#paper)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## News
- **2025-09**: Paper draft cleaned up; Overleaf project added.  
- **2025-09**: Public README and repo structure standardized.

---

## Highlights
- **End-to-end document shadow removal**; preserves strokes and layout for OCR.
- **Background prior** via CBENet, and **shadow-targeted refinement** via BGShadowNet.
- **Transformer backbone** improves robustness to non-uniform illumination and soft shadows.
- Ready-to-extend **training/eval scripts** and **config-driven** experiments.

---

## Repo Structure
The exact layout may vary; this is the intended structure:
```
DISR_ViTs/
├─ src/                       # Training & model code (ViT, CBENet, BGShadowNet, losses, dataloaders)
├─ configs/                   # YAML configs for model & training (dataset paths, hparams)
├─ scripts/                   # Helper scripts (train/eval/infer, dataset prep)
├─ weights/                   # (Optional) Pretrained checkpoints
├─ data/                      # (Optional) Local data root; see Datasets section
├─ paper/                     # Overleaf/LaTeX project of the paper
│  ├─ main.tex
│  └─ images/
├─ results/                   # Saved predictions, metrics, visualizations
├─ requirements.txt           # Python dependencies
└─ README.md
```
> If your repo currently differs, keep this README and adapt folders as needed; the commands below assume `src/` + `configs/` + `scripts/` layout.

---

## Setup

### 1) Environment
- Python **3.9–3.11**
- PyTorch **>= 1.12** (install per your CUDA)
- Recommended: create a fresh virtual environment
```bash
# conda (recommended)
conda create -n disr_vits python=3.10 -y
conda activate disr_vits

# or: venv
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# Windows: .venv\Scripts\activate

# install pytorch (choose your CUDA/CPU at pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install project deps
pip install -r requirements.txt
```

### 2) Optional: Install extras for visualization & OCR checks
```bash
pip install matplotlib opencv-python pillow pytesseract pandas
```

---

## Datasets
We recommend organizing dataset roots under `data/`. This project expects CSV or JSON filelists (train/val/test) with absolute/relative paths.

Commonly used document-shadow datasets include (examples):
- **DocShadow / DocShadow+**
- **AISTD / ISTD**
- **SRD**

> ⚠️ Please comply with the licenses of each dataset. Update your local paths in the YAML under `configs/` (see examples).

Example config snippet (`configs/data_example.yaml`):
```yaml
dataset:
  name: DocShadow
  root: /path/to/data/DocShadow
  csv_train: /path/to/splits/train.csv
  csv_val:   /path/to/splits/val.csv
  csv_test:  /path/to/splits/test.csv
  input_size: [512, 512]
```

To generate filelists, you can adapt a helper like:
```bash
python scripts/make_dataset_csv.py \
  --root /path/to/DocShadow \
  --out  data/splits/docshadow_train.csv \
  --pattern "**/*.jpg"
```

---

## Training
Example single-GPU run (modify for your filenames):
```bash
python scripts/train.py \
  --config configs/model=vit_cbenet_bgshadow.yaml \
  --save_dir results/exp_v1
```

Typical config fields:
```yaml
experiment:
  seed: 42
  save_every: 5
model:
  backbone: vit_tiny          # or vit_small / vit_base
  cbenet: true
  bgshadow: true
loss:
  recon: l1                   # l1 or l2
  ssim: 1.0
  perceptual: 0.0             # set >0 to enable VGG perceptual
optimizer:
  name: adamw
  lr: 2.0e-4
  weight_decay: 0.05
scheduler:
  name: cosine
  warmup_epochs: 5
data:
  ...                         # see Datasets section
train:
  batch_size: 4
  num_workers: 8
```

Multi-GPU (DDP) example:
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/model=vit_cbenet_bgshadow.yaml
```

---

## Evaluation
Evaluate a trained checkpoint on the test split:
```bash
python scripts/eval.py \
  --config configs/model=vit_cbenet_bgshadow.yaml \
  --ckpt weights/exp_v1/best.pt \
  --save_dir results/exp_v1_eval
```
Metrics typically include **PSNR / SSIM**; optional OCR fidelity checks via Tesseract can be added.

---

## Inference
Run on a folder of images and save outputs:
```bash
python scripts/infer.py \
  --ckpt weights/exp_v1/best.pt \
  --input_dir ./samples/inputs \
  --output_dir ./samples/outputs \
  --size 512 512
```
> For scanned PDFs, convert to images first (e.g., `pdfimages` or `poppler`) then run `infer.py`.

---

## Results & Models
| Dataset    | Metric | Baseline A | Baseline B | **DISR_ViTs (Ours)** |
|------------|--------|------------|------------|----------------------|
| DocShadow  | PSNR   |  —         |  —         |  —                   |
| DocShadow  | SSIM   |  —         |  —         |  —                   |
| AISTD/ISTD | PSNR   |  —         |  —         |  —                   |
| AISTD/ISTD | SSIM   |  —         |  —         |  —                   |

- ✅ Please fill the table with your numbers once experiments finish.  
- (Optional) Release checkpoints to `weights/` and link them here.

---

## Reproducibility Notes
- Seed all randomness (`torch`, `numpy`, dataloader workers).
- Log **commit hash**, **config YAML**, and **exact dataset filelists** used.
- Recommend reporting averages over 3 runs for stability.
- Save both **best** and **last** checkpoints; evaluate both.

---

## Paper
The LaTeX source is under `paper/` (IEEEtran). To recompile:
- Open the Overleaf project or compile locally with `pdflatex`/`xelatex` and `bibtex`.
- Replace placeholder BibTeX entries with the correct references for a clean bibliography.

---

## Citation
If you find this work useful, please cite:
```bibtex
@misc{DISR_ViTs_2025,
  title   = {DISR\_ViTs: Document Image Shadow Removal with Vision Transformers},
  author  = {Nguyen Quoc Truong and collaborators},
  year    = {2025},
  howpublished = {\url{<repo-url>}},
  note    = {Code and paper draft}
}
```

---

## License
This repository is released under the **MIT License** (unless otherwise specified by third‑party datasets/models).  
See `LICENSE` for details.

---

## Acknowledgements
- **CBENet** and **BGShadowNet** inspired our background guidance design.
- We also reviewed recent transformer-based shadow removal methods (e.g., ShadowFormer, SpA‑Former, etc.).
- Thanks to the maintainers of PyTorch and the open-source community.

---

### Maintainer
**Nguyễn Quốc Trường** — issues and PRs are welcome!
