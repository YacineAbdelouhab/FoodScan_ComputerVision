import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="FoodScan — Calorie Estimator",
    page_icon="🍽️",
    layout="centered"
)

HF_REPO  = "AbdelouhabYacine/foodscan_models"
IMG_SIZE = 224
DROPOUT  = 0.25
PRED_MIN = 30.0
PRED_MAX = 3600.0
BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k"
FILE     = "convnextv2_base_augmax_full.pth"


# ── MODEL ─────────────────────────────────────────────────────

class ImageOnlyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=False, num_classes=0, global_pool='avg')
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(DROPOUT / 2),
            nn.Linear(128, 1),
        )
    def forward(self, img):
        return self.head(self.backbone(img)).squeeze(1)


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model():
    path  = hf_hub_download(repo_id=HF_REPO, filename=FILE)
    model = ImageOnlyRegressor()
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.eval()
    return model


# ── TRANSFORMS ───────────────────────────────────────────────

def pil_to_tensor(image):
    tfm = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img = np.array(image.convert("RGB"))
    t   = tfm(image=img)['image'].unsqueeze(0)
    return t, torch.flip(t, dims=[3])


# ── PAGE ──────────────────────────────────────────────────────

st.title("🍽️ FoodScan")
st.subheader("Food Calorie Estimator")
st.write("Upload a photo of a food dish and the model will estimate its calorie content.")
st.divider()

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    tensor, tensor_flip = pil_to_tensor(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image uploadée", use_container_width=True)

    with col2:
        try:
            with st.spinner("Inférence en cours..."):
                model = load_model()
                with torch.no_grad():
                    p      = model(tensor)
                    p_flip = model(tensor_flip)
                pred   = float(((p + p_flip) / 2).item())
                result = float(np.clip(np.expm1(pred), PRED_MIN, PRED_MAX))

            st.metric(label="Calories estimées", value=f"{result:.0f} kcal")
            st.caption("Modèle : ConvNextV2-Base AugMax")

        except Exception as e:
            st.error(f"Erreur : {e}")

st.divider()
st.caption(
    "FoodScan Challenge — Deep Learning For Images | "
    "M2 IASD Apprenticeship | Université Paris Dauphine - PSL"
)
