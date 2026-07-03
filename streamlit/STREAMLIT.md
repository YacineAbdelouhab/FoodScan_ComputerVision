# Streamlit App — Build & Deploy Guide

This document explains how to build your FoodScan Streamlit app, test it locally, and deploy it for free on Streamlit Community Cloud.

---

## What Your App Must Do

- Accept a food image upload (JPG or PNG)
- Run inference using your trained model
- Display the predicted calorie estimate clearly
- Be publicly accessible via a URL at presentation time

---

## Step 1 — Export Your Model from Kaggle

At the end of your Kaggle notebook, save your trained model weights:

```python
# At the end of your Kaggle notebook
torch.save(model.state_dict(), "best_model.pt")
```

Then in the Kaggle notebook sidebar:
1. Go to the **Output** tab
2. Download `best_model.pt` to your computer

> **If your model is large (> 500 MB):** The free Streamlit Cloud tier has memory limits. Consider one of the following:
> - Apply PyTorch dynamic quantization to reduce model size:
>
> ```python
> import torch
> model_quantized = torch.quantization.quantize_dynamic(
>     model, {torch.nn.Linear}, dtype=torch.qint8
> )
> torch.save(model_quantized.state_dict(), "best_model_quantized.pt")
> ```
>
> If you encounter any hosting issues, contact `mehyar.mlaweh@dauphine.eu` before the deadline — do not wait.

---


## Step 2 — Complete the Skeleton

Open `app.py` and complete all the `TODO` sections:
- Load your model architecture (copy your model class from Kaggle)
- Load the saved weights
- Implement the preprocessing pipeline (must match exactly what you used during training)
- Run inference and display the result

---

## Step 3 — Test Locally

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`. Test with several food images before deploying.

---

## Step 4 — Deploy on Streamlit Community Cloud

1. Push your completed app to a **public GitHub repository**

```bash
git init
git add .
git commit -m "FoodScan Streamlit app"
git remote add origin https://github.com/your-username/your-repo-name
git push -u origin main
```

2. Go to **https://share.streamlit.io/** and sign in with GitHub

3. Click **"New app"**

4. Fill in:
   - Repository: your GitHub repo
   - Branch: `main`
   - Main file path: `app.py`

5. Click **"Deploy"** — your app will be live at:
   ```
   https://your-username-your-repo-name-app-xxxx.streamlit.app
   ```

6. Copy this URL — this is what you submit by email

---

## Repository Structure for Deployment

Your GitHub repository should look like this:

```
your-repo/
├── app.py                ← your completed Streamlit app
├── requirements.txt      ← all dependencies
└── best_model.pt         ← your exported model weights
```

> **Important:** Do not upload files larger than 100 MB to GitHub directly.
> If your model exceeds this, host the weights on Hugging Face Hub
> and load them programmatically inside `app.py`.

---

*For any issues with deployment, contact `mehyar.mlaweh@dauphine.eu` before the deadline.*
