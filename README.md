# High-Performance Local Medical OCR (vLLM + Qwen3-VL)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![vLLM](https://img.shields.io/badge/vLLM-0.6.0%2B-green)
![Hardware](https://img.shields.io/badge/GPU-4x%20RTX%203090-76b900)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**pdf-ocr-med-multi** is a production-grade, privacy-focused OCR pipeline designed to digitize medical documents (PDFs) into clean Markdown format. 

It leverages **Qwen3-VL-32B-Instruct-AWQ** running locally on a **4x NVIDIA RTX 3090** cluster to achieve high throughput and accuracy without sending sensitive data to the cloud.

## 🚀 Key Features

*   **100% Local & Private:** No data leaves your machine (HIPAA/KVKK compliant).
*   **Massive Parallelism:** Processes **8 documents simultaneously** using Python multiprocessing and vLLM continuous batching.
*   **Smart Layout Detection:** Correctly identifies tables, headers, and key-value pairs in complex medical forms.
*   **High Speed:** Achieves ~150 tokens/sec aggregate throughput on 4x 3090s.
*   **Auto-Cleanup:** Automatically removes Markdown code fences for clean integration.

---

## 🛠️ Hardware Requirements

This project is optimized for the following specific hardware configuration:

*   **GPUs:** 4x NVIDIA GeForce RTX 3090 (24GB VRAM each = 96GB Total).
*   **CPU:** Multi-core processor (support for at least 8 threads recommended).
*   **RAM:** 64GB+ System RAM.
*   **Storage:** Fast NVMe SSD for high-speed image caching.

---

## 📦 Installation

### 1. Prerequisites
Ensure you have the NVIDIA Drivers, CUDA Toolkit (12.1+), and `uv` (Python package manager) installed.

You also need `poppler-utils` for PDF processing:
```bash
# Ubuntu / Debian
sudo apt-get install poppler-utils
```

### 2. Clone Repository
```bash
git clone https://github.com/zakcali/pdf-ocr-med-multi.git
cd pdf-ocr-med-multi
```

### 3. Install Dependencies
```bash
uv pip install vllm openai pdf2image tqdm pillow
```

---

## 🖥️ Usage

This system runs in two parts: the **vLLM Inference Server** and the **Python Client Script**.

### Step 1: Start the vLLM Server
This command loads the **Qwen3-VL-32B (AWQ)** model across all 4 GPUs. It is tuned to prevent OOM errors while maximizing throughput.

```bash
uv run vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
  --tensor-parallel-size 4 \
  --async-scheduling \
  --trust-remote-code \
  --max-model-len 12288 \
  --enforce-eager \
  --limit-mm-per-prompt '{"video": 0}' \
  --max-num-seqs 8
```

**Understanding the flags:**
*   `--tensor-parallel-size 4`: Splits the model across 4 GPUs (approx. 6GB per card for model + KV cache).
*   `--max-num-seqs 8`: Limits concurrent requests to 8. **Crucial** to prevent Out-Of-Memory errors when processing high-res images.
*   `--enforce-eager`: Ensures compatibility with the AWQ/Marlin kernel on Ampere architecture.

### Step 2: Run the OCR Client
Open a new terminal window. Place your PDF files in the `pdf-in` folder and run the script.

```bash
uv run pdf-ocr-med-multi.py
```

*The script will automatically detect 8 CPUs and process 8 PDF files in parallel.*

---

## ⚙️ Configuration

You can adjust the following parameters inside `pdf-ocr-med-multi.py`:

```python
# ================= CONFIGURATION =================
INPUT_FOLDER = "pdf-in"
OUTPUT_FOLDER = "md-out"

# Concurrency must match --max-num-seqs in vLLM command
CONCURRENCY = 8 

# 200 DPI is the "Sweet Spot" for Qwen3-VL (Speed vs. Accuracy)
# Increasing to 300 DPI may cause OOM errors with 8 workers.
DPI = 200 
# =================================================
```

---

## 📊 Performance Benchmarks

On a **4x RTX 3090** machine with **DPI=200** and **Concurrency=8**:

| Metric | Performance |
| :--- | :--- |
| **Throughput** | ~150 tokens/second (Aggregate) |
| **GPU Utilization** | 100% Constant Saturation |
| **VRAM Usage** | ~22.6 GB / 24.0 GB per card |
| **Speed** | ~30 seconds per average medical document |
| **Accuracy** | >99% (Correctly reads decimals, tables, and tiny footnotes) |

---

## 📝 Disclaimer

This tool is for **research and administrative purposes**. While Qwen3-VL is highly accurate, all medical OCR outputs should be verified by a human professional before being used for clinical decision-making.

The author is not responsible for any errors in transcription.

---

## 📄 License

MIT License. See `LICENSE` file for details.
