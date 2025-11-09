# Mulitimodal-Speech-Emotion-Recognition
A  Tensorflow implementation of Speech Emotion Recognition using Audio signals and Text Data


# 🧠 GPU Setup Guide for PyTorch (Windows + Jupyter + Conda/venv)

This guide explains how to set up a GPU-enabled environment for PyTorch on Windows and use it properly inside **Jupyter Notebook**.  
Follow these steps carefully to make sure PyTorch runs on your **NVIDIA GPU** instead of CPU.

---

## ⚙️ Step 1: Check GPU and CUDA Support

Open Command Prompt (or Git Bash) and run:

```bash
nvidia-smi
```

You should see your GPU name and CUDA version, for example:

```
CUDA Version: 12.6
GPU Name: NVIDIA GeForce RTX 2080 SUPER
```

---

## 🧩 Step 2: Create a Python Environment (Optional but Recommended)

### 🅰️ If using Conda:

```bash
conda create -n ml python=3.10 -y
conda activate ml
```

### 🅱️ If using virtualenv:

```bash
python -m venv ml
ml\Scripts\activate
```

---

## 🚀 Step 3: Install PyTorch with CUDA Support

Check which CUDA version your GPU supports (from `nvidia-smi`).

Then install the matching PyTorch build.  
Example for **CUDA 12.1**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🧠 Step 4: Verify GPU Availability in Terminal

Run the following test code:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

✅ Expected Output:

```
2.5.1+cu121
True
NVIDIA GeForce RTX 2080 SUPER
```

If `True` appears, your GPU is working with PyTorch.

---

## 🧩 Step 5: Add the Environment to Jupyter Notebook

Install Jupyter kernel support in your environment:

```bash
pip install ipykernel
python -m ipykernel install --user --name=ml --display-name "Python (ml GPU)"
```

Now open **Jupyter Notebook / VS Code**,  
and select the kernel:

```
Kernel → Change Kernel → Python (ml GPU)
```

---

## 🧪 Step 6: Test GPU Inside Jupyter Notebook

Create a new notebook cell and run:

```python
import torch

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    x = torch.randn(10000, 10000).to("cuda")
    y = torch.randn(10000, 10000).to("cuda")
    z = torch.matmul(x, y)
    print("🔥 Matrix multiplication successful on GPU!")
else:
    print("❌ Still running on CPU.")
```

✅ **Expected Output:**

```
PyTorch version: 2.5.1+cu121
✅ GPU detected: NVIDIA GeForce RTX 2080 SUPER
🔥 Matrix multiplication successful on GPU!
```

---

## 🧹 Step 7: (Optional) Clean Up Unused Kernels

To remove any old kernel (e.g., CPU-only):

```bash
jupyter kernelspec list
jupyter kernelspec remove <kernel_name>
```

---

## 🏁 Summary

| Task                       | Status |
| -------------------------- | ------ |
| GPU Detected               | ✅     |
| PyTorch CUDA Version       | ✅     |
| Jupyter Notebook using GPU | ✅     |
| Matrix Operations on GPU   | ✅     |

---

### 💡 Author

**Rakibul Hassan**  
Port City International University  
Machine Learning & AI Enthusiast
# A-Unified-Multimodal-Deep-Learning-Architecture-for-Multidimensional-Human-Emotion-Recognition" 