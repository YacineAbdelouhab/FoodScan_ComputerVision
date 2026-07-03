import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from huggingface_hub import hf_hub_download, snapshot_download

# ──────────────────────────────────────────────────────────────
#  FOODSCAN — CALORIE ESTIMATOR
# ──────────────────────────────────────────────────────────────

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
TEXT_DIM = 768

MODELS = {
    "ConvNextV2-AugMax (défaut)": {
        "file"    : "convnextv2_base_augmax_quantized.pth",
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k",
        "type"    : "imgonly",
        "quantize": True,
        "img_size": False,
        "desc"    : "ConvNextV2-Base, augmentation agressive, quantifié int8 (100 MB). Rapide.",
    },
    "ConvNextV2-MM": {
        "file"    : "convnextv2_base_mm_full.pth",
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k",
        "type"    : "mm",
        "quantize": False,
        "img_size": False,
        "desc"    : "ConvNextV2-Base + description texte (mpnet 768-dim).",
    },
}
DEFAULT_MODEL = "ConvNextV2-AugMax (défaut)"


# ── MODEL BUILDERS ────────────────────────────────────────────

def build_model(cfg):
    kwargs = dict(pretrained=False, num_classes=0, global_pool='avg')
    if cfg["img_size"]:
        kwargs['img_size'] = IMG_SIZE
    backbone = timm.create_model(cfg["backbone"], **kwargs)
    d = backbone.num_features

    if cfg["type"] == "mm":
        fused = d + TEXT_DIM
        head  = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 128),   nn.GELU(), nn.Dropout(DROPOUT / 2),
            nn.Linear(128, 1),
        )
        class MM(nn.Module):
            def __init__(self): super().__init__(); self.backbone=backbone; self.head=head
            def forward(self, img, emb): return self.head(torch.cat([self.backbone(img), emb], 1)).squeeze(1)
        model = MM()
    else:
        head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(DROPOUT / 2),
            nn.Linear(128, 1),
        )
        class IO(nn.Module):
            def __init__(self): super().__init__(); self.backbone=backbone; self.head=head
            def forward(self, img): return self.head(self.backbone(img)).squeeze(1)
        model = IO()

    return model


# ── TRANSFORMS ───────────────────────────────────────────────

def get_tfm():
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def pil_to_tensor(image):
    img = np.array(image.convert("RGB"))
    t   = get_tfm()(image=img)['image'].unsqueeze(0)
    return t, torch.flip(t, dims=[3])


# ── MODEL LOADING ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle par défaut...")
def load_default():
    cfg   = MODELS[DEFAULT_MODEL]
    path  = hf_hub_download(repo_id=HF_REPO, filename=cfg["file"])
    model = build_model(cfg)
    model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.eval()
    return model

@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model_cached(name):
    cfg   = MODELS[name]
    path  = hf_hub_download(repo_id=HF_REPO, filename=cfg["file"])
    model = build_model(cfg)
    if cfg["quantize"]:
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    ckpt  = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    model.eval()
    return model

@st.cache_resource(show_spinner="Chargement du modèle d'embedding...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    import os
    model_path = snapshot_download(repo_id=HF_REPO, allow_patterns="all-mpnet-base-v2/*")
    return SentenceTransformer(os.path.join(model_path, "all-mpnet-base-v2"), device='cpu')

def embed_text(description: str) -> torch.Tensor:
    emb = load_embedder().encode([description], normalize_embeddings=True).astype('float32')[0]
    return torch.tensor(emb).unsqueeze(0)

def predict(model, cfg, tensor, tensor_flip, emb=None) -> float:
    with torch.no_grad():
        if cfg["type"] == "mm":
            emb = emb if emb is not None else torch.zeros(1, TEXT_DIM)
            p, p_flip = model(tensor, emb), model(tensor_flip, emb)
        else:
            p, p_flip = model(tensor), model(tensor_flip)
    return float(np.clip(np.expm1(float(((p + p_flip) / 2).item())), PRED_MIN, PRED_MAX))


# ── PAGE ──────────────────────────────────────────────────────

st.title("🍽️ FoodScan")
st.subheader("Food Calorie Estimator")
st.write("Upload a photo of a food dish and the model will estimate its calorie content.")
st.divider()

# AugMax préchargé au démarrage (100MB seulement)
load_default()

choice = st.selectbox("Modèle", options=list(MODELS.keys()), index=0)
st.caption(MODELS[choice]["desc"])

is_mm = MODELS[choice]["type"] == "mm"

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

description = None
if uploaded_file and is_mm:
    description = st.text_area(
        "Décrivez le contenu de l'assiette",
        placeholder="Ex: pasta bolognaise avec parmesan et basilic...",
        help="La description améliore la précision du modèle multimodal."
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    tensor, tensor_flip = pil_to_tensor(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image uploadée", use_container_width=True)

    with col2:
        try:
            with st.spinner("Inférence en cours..."):
                load_default()  # toujours en cache
                model = load_default() if choice == DEFAULT_MODEL else load_model_cached(choice)

                emb = None
                if is_mm and description and description.strip():
                    with st.spinner("Encodage de la description..."):
                        emb = embed_text(description.strip())

                result = predict(model, MODELS[choice], tensor, tensor_flip, emb)

            st.metric(label="Calories estimées", value=f"{result:.0f} kcal")
            if is_mm and not (description and description.strip()):
                st.info("💡 Ajoutez une description pour améliorer la précision.")

        except Exception as e:
            st.error(f"Erreur : {e}")

st.divider()
st.caption(
    "FoodScan Challenge — Deep Learning For Images | "
    "M2 IASD Apprenticeship | Université Paris Dauphine - PSL"
)
