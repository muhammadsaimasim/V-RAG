# Visual RAG — CIFAR-10 + CLIP + FAISS

A local image similarity search engine. Provide any image as input and retrieve the **top 10 most visually similar images** from a CIFAR-10 database — no cloud, no GPU required.

---

## What It Does

1. Loads **1 500 images** from the CIFAR-10 dataset
2. Encodes every image into a **512-dimensional vector** using CLIP
3. Stores all vectors in a **FAISS similarity index**
4. Takes **any image you provide** as a query
5. Returns the **top 10 most similar images** with similarity scores
6. Displays results as a **labelled matplotlib grid** and saves to `rag_results.png`

---

## Project Structure

```
visual_rag/
├── visual_rag_v2.py       ← main script
├── requirements.txt       ← all dependencies
├── .gitignore             ← files excluded from git
├── README.md              ← this file
├── my_image.jpg           ← your query image (you provide this)
│
├── cifar_data/            ← CIFAR-10 dataset (auto-downloaded, ~170 MB)
└── rag_results.png        ← output results grid (generated on run)
```

---

## Requirements

- Python 3.8 or higher
- No GPU needed — runs fully on CPU

---

## Setup

**1. Clone or download the project**
```bash
git clone <your-repo-url>
cd visual_rag
```

**2. Create a virtual environment**
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

**1. Add your query image**

Place any image file (JPG, PNG, BMP, or WEBP) in the project folder.

**2. Set the query image path**

Open `visual_rag_v2.py` and update this one line near the bottom of `main()`:

```python
QUERY_IMAGE_PATH = "my_image.jpg"   # ← change to your filename
```

**3. Run**

```bash
python visual_rag_v2.py
```

---

## Expected Output

**Terminal:**
```
==================================================
  Visual RAG  —  CIFAR-10 + CLIP + FAISS
==================================================

[1/5] Loading CIFAR-10 (1500 images)...
      Loaded 1500 images.

[2/5] Loading CLIP model...
      Using device: cpu

[3/5] Generating embeddings...
      1500/1500

[4/5] Building FAISS index...
      Vectors stored: 1500

[5/5] Querying with image: 'my_image.jpg'...
      Top 10 results:
       #1  idx= 234  score=0.9312  class=dog
       #2  idx= 891  score=0.9187  class=dog
       ...

 Done!
```

**Visual:** A matplotlib window opens showing your query image alongside the 10 most similar CIFAR-10 images. The grid is also saved as `rag_results.png`.

> **Note:** The first run downloads CIFAR-10 (~170 MB) and CLIP model weights (~350 MB). This takes a few minutes. Every run after that is fast — files are cached locally.

---

## Configuration

All settings are in the `CONFIG` block at the top of `visual_rag_v2.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_IMAGES` | `1500` | Number of CIFAR-10 images to index (1000–2000 recommended) |
| `BATCH_SIZE` | `64` | Images processed per batch — reduce to `32` if you get memory errors |
| `TOP_K` | `10` | Number of similar images to retrieve |
| `CLIP_MODEL` | `ViT-B-32` | CLIP architecture — ViT-B-32 is the fastest for CPU |
| `PRETRAINED` | `openai` | Pre-trained weights source |
| `DATA_DIR` | `./cifar_data` | Where CIFAR-10 is downloaded and cached |

---

## How It Works

```
Your image file
      │
      ▼  PIL open + convert RGB
      │
      ▼  CLIP preprocess (resize 224×224, normalise)
      │
      ▼  CLIP ViT-B-32 encode_image()
      │
      ▼  L2 normalise → 512D unit vector
      │
      ▼  FAISS IndexFlatIP search (cosine similarity)
      │
      ▼  Top 10 indices + scores
      │
      ▼  Fetch images → display grid
```

**CLIP** (Contrastive Language–Image Pretraining) converts images into 512-number vectors where visually similar images produce similar vectors.

**FAISS** (Facebook AI Similarity Search) stores all 1 500 vectors and finds the nearest ones to any query vector in milliseconds.

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | PyTorch — deep learning framework |
| `torchvision` | CIFAR-10 dataset loader and image transforms |
| `open_clip_torch` | CLIP model — image embedding generation |
| `faiss-cpu` | Vector similarity search index |
| `matplotlib` | Results visualisation |
| `numpy` | Numerical arrays and matrix operations |
| `Pillow` | Image file loading and format conversion |

---

## CIFAR-10 Classes

The dataset contains 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

---

## Troubleshooting

**PowerShell blocks venv activation**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Memory error during embedding generation**
Reduce `BATCH_SIZE` from `64` to `32` in the CONFIG block.

**Query image not found error**
Make sure your image file is in the same folder as `visual_rag_v2.py` and the filename in `QUERY_IMAGE_PATH` matches exactly (including the extension).

**Slow on first run**
Normal — CIFAR-10 and CLIP weights are downloading. Subsequent runs skip this step entirely.

---

## License

This project is for educational purposes. CIFAR-10 dataset credit: [Alex Krizhevsky, University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html). CLIP model credit: [OpenAI](https://github.com/openai/CLIP).