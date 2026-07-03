# FoodScan Challenge

Final project for the Deep Learning For Images course (M2 IASD Apprenticeship, Université Paris Dauphine – PSL).

Goal: predict the number of calories in a dish from a single photo. It's a Kaggle competition: https://www.kaggle.com/competitions/m2-food-calorie-estimation

Streamlit app: https://foodscancomputervision.streamlit.app

## What's in this repo

- `code/code_kaggle/` : the Kaggle notebooks used for training (links in `url_kaggle.txt`)
- `code/code_best_config/` : the training config we kept in the end (best results)
- `code/code_collab/` : early tests on Colab before moving to Kaggle
- `streamlit/` : the Streamlit app (app.py + requirements.txt)

## What we did

Calories are very spread out (from 50 to over 3000 kcal), so we train on `log(calories)` instead of raw calories, and convert back at the end.

We tried several pretrained vision backbones with a small regression head on top. Some models only take the image, others also take a text description of the dish, combined before the head.

The final model is a blend/ensemble of several models, weights picked by looking at validation MAE (OOF).

The Kaggle notebooks used for training are linked in `url_kaggle.txt`.

## Streamlit

The app lets you upload a photo and pick between the image-only model and the multimodal model (with an optional text description). Weights are downloaded from Hugging Face at inference time (to avoid running out of RAM on Streamlit Cloud).

To run locally:

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```