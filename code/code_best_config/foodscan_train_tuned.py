# # FoodScan Challenge - High-Performance Calorie Regression
# 
# This notebook trains strong ImageNet-pretrained vision models for food calorie regression. It is designed to run both locally and on Kaggle, with strict CUDA usage, stratified K-Fold validation, log-target regression, mixed precision, test-time augmentation, and multi-model ensembling.
# 
# The default configuration is intentionally competitive rather than minimal. For a quick smoke test, set `MODE = "debug"`. For the final Kaggle run, set `MODE = "final"` and train all folds/experiments.

# Section: Optional Environment Setup


# Optional Kaggle install cell.
# Uncomment this if Kaggle does not already have these packages.
# !pip install -q timm albumentations opencv-python-headless


# Section: Imports And Global Display Settings


import os
import gc
import json
import math
import random
import time
import warnings
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from IPython.display import display
except Exception:
    # Fallback for plain Python execution outside notebooks.
    def display(obj):
        print(obj)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)


# Section: Experiment And Runtime Configuration


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    img_size: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    epochs: int
    patience: int
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    min_lr: float = 1e-6
    loss_beta: float = 0.08
    pretrained: bool = True
    grad_checkpointing: bool = False

@dataclass
class GlobalConfig:
    seed: int = 42
    n_folds: int = 5
    num_workers: int = 4
    target_col: str = "calories"
    pred_clip_min: float = 30.0
    pred_clip_max: float = 3600.0
    tta_flips: bool = True
    out_dir: str = "outputs_foodscan_duo_tuned"
    strict_cuda: bool = True
    label_noise_std: float = 0.0
    progress_bars: bool = False  # Keep False in VS Code Remote to avoid notebook renderer crashes.
    channels_last: bool = True

# Modes:
# - "debug": fast sanity check on one fold and few epochs.
# - "strong": one strong model, all folds.
# - "final": multi-model ensemble, all folds, more epochs.
MODE = "final"

gcfg = GlobalConfig()

# Select folds and experiment grid from the chosen run mode.
if MODE == "debug":
    RUN_FOLDS = (0,)
    EXPERIMENTS = [
        ExperimentConfig(
            name="convnext_base_384_debug",
            model_name="convnext_base.fb_in22k_ft_in1k",
            img_size=384,
            batch_size=4,
            lr=2e-4,
            weight_decay=5e-2,
            dropout=0.20,
            epochs=2,
            patience=2,
        )
    ]
elif MODE == "strong":
    RUN_FOLDS = (0, 1, 2, 3, 4)
    EXPERIMENTS = [
        ExperimentConfig(
            name="convnext_base_384",
            model_name="convnext_base.fb_in22k_ft_in1k",
            img_size=384,
            batch_size=8,
            lr=2e-4,
            weight_decay=5e-2,
            dropout=0.20,
            epochs=35,
            patience=8,
        )
    ]
elif MODE == "final":
    RUN_FOLDS = (0, 1, 2, 3, 4)
    EXPERIMENTS = [
        ExperimentConfig(
            name="convnext_large_448_tuned_a",
            model_name="convnext_large.fb_in22k_ft_in1k_384",
            img_size=448,
            batch_size=4,
            lr=1.4e-4,
            weight_decay=5e-2,
            dropout=0.25,
            epochs=100,
            patience=16,
            grad_accum=2,
            grad_checkpointing=True,
        ),
        ExperimentConfig(
            name="convnext_large_448_tuned_b",
            model_name="convnext_large.fb_in22k_ft_in1k_384",
            img_size=448,
            batch_size=4,
            lr=1.1e-4,
            weight_decay=7e-2,
            dropout=0.30,
            epochs=120,
            patience=20,
            grad_accum=2,
            grad_checkpointing=True,
        ),
        ExperimentConfig(
            name="swinv2_large_384_tuned_a",
            model_name="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
            img_size=384,
            batch_size=3,
            lr=9.0e-5,
            weight_decay=5e-2,
            dropout=0.28,
            epochs=100,
            patience=16,
            grad_accum=3,
            grad_checkpointing=True,
        ),
        ExperimentConfig(
            name="swinv2_large_384_tuned_b",
            model_name="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
            img_size=384,
            batch_size=3,
            lr=8.5e-5,
            weight_decay=6e-2,
            dropout=0.32,
            epochs=120,
            patience=20,
            grad_accum=3,
            grad_checkpointing=True,
        ),
        ExperimentConfig(
            name="convnext_large_448_tuned_c",
            model_name="convnext_large.fb_in22k_ft_in1k_384",
            img_size=448,
            batch_size=4,
            lr=1.25e-4,
            weight_decay=6e-2,
            dropout=0.27,
            epochs=140,
            patience=24,
            grad_accum=2,
            grad_checkpointing=True,
        ),
        ExperimentConfig(
            name="swinv2_large_384_tuned_c",
            model_name="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
            img_size=384,
            batch_size=3,
            lr=8.0e-5,
            weight_decay=7e-2,
            dropout=0.35,
            epochs=140,
            patience=24,
            grad_accum=3,
            grad_checkpointing=True,
        )
    ]
else:
    raise ValueError(f"Unknown MODE: {MODE}")

LOCAL_ROOT = Path(".")
TRAIN_CSV = LOCAL_ROOT / "train_labels.csv"
TEST_CSV = LOCAL_ROOT / "test_ids.csv"
TRAIN_IMG_DIR = LOCAL_ROOT / "train" / "images"
TEST_IMG_DIR = LOCAL_ROOT / "test" / "images"

# Kaggle competition paths.
# KAGGLE_ROOT = Path("/kaggle/input/competitions/m2-food-calorie-estimation")
# TRAIN_CSV = KAGGLE_ROOT / "train_labels.csv"
# TEST_CSV = KAGGLE_ROOT / "test_ids.csv"
# TRAIN_IMG_DIR = KAGGLE_ROOT / "train" / "images"
# TEST_IMG_DIR = KAGGLE_ROOT / "test" / "images"
# gcfg.out_dir = "/kaggle/working/outputs_foodscan"

Path(gcfg.out_dir).mkdir(parents=True, exist_ok=True)

def make_unique_run_dir(base_dir: str, mode: str, seed: int) -> str:
    # Create a unique run folder so concurrent runs never overwrite each other.
    base = Path(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"run_{mode}_{stamp}_s{seed}"
    idx = 1
    while candidate.exists():
        idx += 1
        candidate = base / f"run_{mode}_{stamp}_s{seed}_v{idx}"
    return str(candidate)

gcfg.out_dir = make_unique_run_dir(gcfg.out_dir, MODE, gcfg.seed)
Path(gcfg.out_dir).mkdir(parents=True, exist_ok=False)

print("Mode:", MODE)
print("Folds:", RUN_FOLDS)
print("Output directory:", gcfg.out_dir)
print("Experiments:")
for exp in EXPERIMENTS:
    print(asdict(exp))


# Section: Reproducibility And Device Initialization


def seed_everything(seed: int = 42) -> None:
    # Seed all relevant RNGs for reproducible CV splits and training behavior.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

seed_everything(gcfg.seed)

if gcfg.strict_cuda:
    assert torch.cuda.is_available(), "CUDA is not available. Enable a Kaggle GPU or use a local CUDA PyTorch build."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("timm:", timm.__version__)


# Section: Dataset Loading And Path Validation


train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Store absolute image paths once to avoid repeated path joins later.
train_df["path"] = train_df["filename"].apply(lambda x: str(TRAIN_IMG_DIR / x))
test_df["path"] = test_df["filename"].apply(lambda x: str(TEST_IMG_DIR / x))

# Fail fast if any data path is broken, before expensive model initialization.
missing_train = train_df.loc[~train_df["path"].map(lambda p: Path(p).exists())]
missing_test = test_df.loc[~test_df["path"].map(lambda p: Path(p).exists())]
assert missing_train.empty, missing_train.head()
assert missing_test.empty, missing_test.head()
assert len(test_df) == 547, f"Unexpected test size: {len(test_df)}"

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
display(train_df.head())
display(train_df[gcfg.target_col].describe())


# Section: Basic Target Distribution Plots


fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(train_df[gcfg.target_col], bins=70, ax=axes[0])
axes[0].set_title("Calorie distribution")
sns.histplot(np.log1p(train_df[gcfg.target_col]), bins=70, ax=axes[1])
axes[1].set_title("log1p calorie distribution")
plt.tight_layout()
plt.show()


# Section: Visual Sample Inspection


def show_samples(df: pd.DataFrame, n: int = 12) -> None:
    sample = df.sample(n=min(n, len(df)), random_state=gcfg.seed).reset_index(drop=True)
    cols = 4
    rows = math.ceil(len(sample) / cols)
    plt.figure(figsize=(14, 3.5 * rows))
    for i, row in sample.iterrows():
        img = cv2.imread(row.path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{row.image_id} | {row.calories:.1f} kcal")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(train_df, n=12)


# Section: Fold Construction


train_df = train_df.copy()
# Train on log-target to stabilize optimization under heavy-tailed calorie values.
train_df["target_log"] = np.log1p(train_df[gcfg.target_col].astype(float))
train_df["calorie_bin"] = pd.qcut(train_df[gcfg.target_col], q=20, labels=False, duplicates="drop")
train_df["fold"] = -1

# Stratification by calorie quantile helps keep fold distributions consistent.
skf = StratifiedKFold(n_splits=gcfg.n_folds, shuffle=True, random_state=gcfg.seed)
for fold, (_, valid_idx) in enumerate(skf.split(train_df, train_df["calorie_bin"])):
    train_df.loc[valid_idx, "fold"] = fold

fold_stats = train_df.groupby("fold")[gcfg.target_col].agg(["count", "mean", "std", "min", "max"])
display(fold_stats)


# Section: Data Augmentation Pipelines


def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size + 96),
        A.PadIfNeeded(min_height=img_size + 96, min_width=img_size + 96, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.65, 1.0), ratio=(0.80, 1.20), interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.84, 1.16), translate_percent=(-0.035, 0.035), rotate=(-15, 15), border_mode=cv2.BORDER_REFLECT_101, p=0.60),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20),
            A.HueSaturationValue(hue_shift_limit=7, sat_shift_limit=16, val_shift_limit=12),
            A.CLAHE(clip_limit=2.0),
        ], p=0.50),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=3),
            A.ImageCompression(quality_range=(72, 100)),
        ], p=0.22),
        A.CoarseDropout(num_holes_range=(1, 7), hole_height_range=(0.035, 0.12), hole_width_range=(0.035, 0.12), fill=0, p=0.20),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# Section: Dataset And DataLoader Utilities


class FoodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: A.Compose = None, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(row.path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(row.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        if self.is_train:
            target = float(row.target_log)
            if gcfg.label_noise_std > 0:
                target += np.random.normal(0.0, gcfg.label_noise_std)
            return image, torch.tensor(target, dtype=torch.float32)

        return image, row.image_id


def make_loader(df: pd.DataFrame, transforms: A.Compose, exp: ExperimentConfig, is_train: bool, shuffle: bool, drop_last: bool = False) -> DataLoader:
    # Keep DataLoader creation centralized so worker/pin-memory behavior stays consistent.
    dataset = FoodDataset(df, transforms=transforms, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=exp.batch_size,
        shuffle=shuffle,
        num_workers=gcfg.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=gcfg.num_workers > 0,
    )


# Section: Model Definition And Builder


class FoodRegressor(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Dropout(dropout),
            nn.Linear(n_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.mean(dim=tuple(range(2, features.ndim)))
        return self.head(features).squeeze(1)


def build_model(exp: ExperimentConfig) -> nn.Module:
    # Build backbone + regression head and apply optional memory optimizations.
    model = FoodRegressor(exp.model_name, pretrained=exp.pretrained, dropout=exp.dropout)
    if exp.grad_checkpointing and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(enable=True)
    model = model.to(device)
    if gcfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


# Section: Metrics, Post-Processing, And Optimizer


def calories_from_log(values: np.ndarray) -> np.ndarray:
    values = np.expm1(values)
    return np.clip(values, gcfg.pred_clip_min, gcfg.pred_clip_max)


def mae_from_log(pred_log: np.ndarray, true_log: np.ndarray) -> float:
    # Report MAE in original calorie space, not in log space.
    pred = calories_from_log(pred_log)
    true = np.expm1(true_log)
    return mean_absolute_error(true, pred)


def get_optimizer(model: nn.Module, exp: ExperimentConfig) -> torch.optim.Optimizer:
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": exp.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=exp.lr,
    )


def checkpoint_path(exp: ExperimentConfig, fold: int) -> Path:
    return Path(gcfg.out_dir) / exp.name / f"fold{fold}.pth"


# Section: Train/Validation Loop Helpers


def iterate_batches(loader, desc: str):
    if gcfg.progress_bars:
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=5)
    return loader


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, exp: ExperimentConfig) -> float:
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(iterate_batches(loader, "train")):
        images = images.to(device, non_blocking=True)
        if gcfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, targets) / exp.grad_accum

        scaler.scale(loss).backward()

        # Gradient accumulation simulates a larger effective batch size.
        if (step + 1) % exp.grad_accum == 0:
            if exp.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), exp.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        losses.append(float(loss.item() * exp.grad_accum))

    return float(np.mean(losses))


@torch.no_grad()
def validate_one_epoch(model, loader, criterion) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds, targets_all = [], [], []

    for images, targets in iterate_batches(loader, "valid"):
        images = images.to(device, non_blocking=True)
        if gcfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, targets)

        losses.append(float(loss.item()))
        preds.append(outputs.detach().float().cpu().numpy())
        targets_all.append(targets.detach().float().cpu().numpy())

    preds = np.concatenate(preds)
    targets_all = np.concatenate(targets_all)
    valid_mae = mae_from_log(preds, targets_all)
    return float(np.mean(losses)), valid_mae, preds, targets_all


# Section: Per-Fold Training Routine


def train_fold(exp: ExperimentConfig, fold: int) -> Dict:
    print(f"\n===== Experiment {exp.name} | Fold {fold}/{gcfg.n_folds - 1} =====")
    exp_dir = Path(gcfg.out_dir) / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_part = train_df[train_df.fold != fold].reset_index(drop=True)
    valid_part = train_df[train_df.fold == fold].reset_index(drop=True)

    train_loader = make_loader(train_part, get_train_transforms(exp.img_size), exp, is_train=True, shuffle=True, drop_last=True)
    valid_loader = make_loader(valid_part, get_valid_transforms(exp.img_size), exp, is_train=True, shuffle=False, drop_last=False)

    model = build_model(exp)
    optimizer = get_optimizer(model, exp)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / exp.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps_per_epoch * exp.epochs,
        eta_min=exp.min_lr,
    )
    scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())
    criterion = nn.SmoothL1Loss(beta=exp.loss_beta)

    best_mae = float("inf")
    best_epoch = -1
    wait = 0
    history = []
    best_path = checkpoint_path(exp, fold)

    # Early stopping runs on validation MAE to reduce overfitting and wasted compute.
    for epoch in range(1, exp.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, exp)
        valid_loss, valid_mae, valid_preds_log, valid_targets_log = validate_one_epoch(model, valid_loader, criterion)
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "experiment": exp.name,
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_mae": valid_mae,
            "lr": lr,
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train {train_loss:.5f} | valid {valid_loss:.5f} | "
            f"MAE {valid_mae:.3f} | lr {lr:.2e} | {elapsed:.0f}s"
        )

        if valid_mae < best_mae:
            best_mae = valid_mae
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "global_config": asdict(gcfg),
                    "experiment_config": asdict(exp),
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_mae": best_mae,
                },
                best_path,
            )
            assert len(valid_preds_log) == len(valid_part), f"Validation prediction length mismatch: {len(valid_preds_log)} vs {len(valid_part)}"
            oof = valid_part[["image_id", "filename", gcfg.target_col, "fold"]].copy()
            oof["pred_log"] = valid_preds_log
            oof["predicted_calories"] = calories_from_log(valid_preds_log)
            oof.to_csv(exp_dir / f"oof_fold{fold}.csv", index=False)
            print(f"saved best checkpoint: {best_path} | MAE {best_mae:.3f}")
        else:
            wait += 1
            if wait >= exp.patience:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch}, best MAE {best_mae:.3f}")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / f"history_fold{fold}.csv", index=False)

    del model, optimizer, scheduler, scaler, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "experiment": exp.name,
        "fold": fold,
        "best_mae": best_mae,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
    }


# Section: Full Experiment Training Execution


all_results = []
for exp in EXPERIMENTS:
    print(f"\n########## Starting experiment: {exp.name} ##########")
    exp_dir = Path(gcfg.out_dir) / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"global": asdict(gcfg), "experiment": asdict(exp), "mode": MODE, "folds": list(RUN_FOLDS)}, f, indent=2)
    for fold in RUN_FOLDS:
        result = train_fold(exp, fold)
        all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df.to_csv(Path(gcfg.out_dir) / "training_results.csv", index=False)
display(results_df)
print("Mean CV by experiment:")
display(results_df.groupby("experiment")["best_mae"].agg(["mean", "std", "min", "max"]))


# Section: OOF Aggregation And Per-Experiment Scoring


def load_oof_predictions() -> pd.DataFrame:
    # Aggregate per-fold OOF files to evaluate each experiment and blend quality.
    frames = []
    for exp in EXPERIMENTS:
        exp_dir = Path(gcfg.out_dir) / exp.name
        for fold in RUN_FOLDS:
            path = exp_dir / f"oof_fold{fold}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["experiment"] = exp.name
                frames.append(df)
    assert frames, "No OOF files found. Train at least one fold first."
    return pd.concat(frames, ignore_index=True)

oof_long = load_oof_predictions()
display(oof_long.head())

for exp_name, part in oof_long.groupby("experiment"):
    score = mean_absolute_error(part[gcfg.target_col], part["predicted_calories"])
    print(f"{exp_name} OOF MAE: {score:.3f}")

# Equal-weight ensemble on the OOF rows available for each image.
oof_wide = oof_long.pivot_table(index="image_id", columns="experiment", values="pred_log", aggfunc="mean")
oof_truth = train_df.set_index("image_id").loc[oof_wide.index, gcfg.target_col]
oof_ensemble = calories_from_log(oof_wide.mean(axis=1).values)
print("Equal-weight ensemble OOF MAE:", mean_absolute_error(oof_truth.values, oof_ensemble))


# Section: Checkpoint Inference Helper


@torch.no_grad()
def predict_checkpoint(exp: ExperimentConfig, fold: int, loader: DataLoader) -> Tuple[List[str], np.ndarray]:
    path = checkpoint_path(exp, fold)
    assert path.exists(), f"Missing checkpoint: {path}"

    model = build_model(exp)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_ids_all, preds_all = [], []
    # Predict in log space and optionally average with horizontal-flip TTA.
    for images, image_ids in iterate_batches(loader, f"predict {exp.name} fold {fold}"):
        images = images.to(device, non_blocking=True)
        if gcfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        batch_preds = []
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            batch_preds.append(model(images).detach().float().cpu().numpy())
            if gcfg.tta_flips:
                batch_preds.append(model(torch.flip(images, dims=[3])).detach().float().cpu().numpy())
        preds_all.append(np.mean(batch_preds, axis=0))
        image_ids_all.extend(list(image_ids))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return image_ids_all, np.concatenate(preds_all)


# Section: Test Predictions, Ensembling, And Blend Search


all_test_pred_logs = []
image_ids_reference = None
prediction_records = []
test_pred_log_map = {}

for exp in EXPERIMENTS:
    test_loader = make_loader(test_df, get_valid_transforms(exp.img_size), exp, is_train=False, shuffle=False, drop_last=False)
    exp_fold_preds = []
    for fold in RUN_FOLDS:
        path = checkpoint_path(exp, fold)
        if not path.exists():
            print(f"Skipping missing checkpoint: {path}")
            continue
        image_ids, pred_log = predict_checkpoint(exp, fold, test_loader)
        if image_ids_reference is None:
            image_ids_reference = image_ids
        else:
            assert image_ids_reference == image_ids
        exp_fold_preds.append(pred_log)
        prediction_records.append({"experiment": exp.name, "fold": fold, "checkpoint": str(path)})

    if exp_fold_preds:
        exp_pred_log = np.mean(exp_fold_preds, axis=0)
        all_test_pred_logs.append(exp_pred_log)
        test_pred_log_map[exp.name] = exp_pred_log
        exp_submission = pd.DataFrame({
            "image_id": image_ids_reference,
            "predicted_calories": calories_from_log(exp_pred_log),
        })
        exp_submission = test_df[["image_id"]].merge(exp_submission, on="image_id", how="left")
        exp_submission.to_csv(Path(gcfg.out_dir) / f"submission_{exp.name}.csv", index=False)

assert all_test_pred_logs, "No test predictions were created. Train checkpoints first."
ensemble_pred_log = np.mean(all_test_pred_logs, axis=0)
ensemble_pred = calories_from_log(ensemble_pred_log)

submission = pd.DataFrame({
    "image_id": image_ids_reference,
    "predicted_calories": ensemble_pred,
})
submission = test_df[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["predicted_calories"].notna().all()
assert len(submission) == 547

# Save only inside this run-specific directory to avoid any overwrite.
run_submission_path = Path(gcfg.out_dir) / "submission_equal_mean.csv"
submission.to_csv(run_submission_path, index=False)

# Build weighted blends between ConvNeXt and Swin variants using OOF guidance.
# Search is performed on OOF predictions, then transferred to test predictions.
oof_map = {
    exp_name: part.set_index("image_id")["pred_log"]
    for exp_name, part in oof_long.groupby("experiment")
}
available_experiments = [name for name in test_pred_log_map.keys() if name in oof_map]
convnext_experiments = [name for name in available_experiments if "convnext" in name]
swin_experiments = [name for name in available_experiments if "swinv2" in name]

blend_records = []
best_blend = None

for exp_name in available_experiments:
    exp_ids = oof_map[exp_name].index
    y_true = train_df.set_index("image_id").loc[exp_ids, gcfg.target_col].values
    single_mae = mean_absolute_error(y_true, calories_from_log(oof_map[exp_name].values))
    blend_records.append(
        {
            "blend_type": "single",
            "convnext_exp": exp_name if "convnext" in exp_name else "",
            "swin_exp": exp_name if "swinv2" in exp_name else "",
            "w_convnext": 1.0 if "convnext" in exp_name else 0.0,
            "w_swin": 1.0 if "swinv2" in exp_name else 0.0,
            "oof_mae": float(single_mae),
        }
    )

for conv_name in convnext_experiments:
    for swin_name in swin_experiments:
        common_ids = oof_map[conv_name].index.intersection(oof_map[swin_name].index)
        y_true = train_df.set_index("image_id").loc[common_ids, gcfg.target_col].values
        conv_oof = oof_map[conv_name].loc[common_ids].values
        swin_oof = oof_map[swin_name].loc[common_ids].values

        for w_conv in np.round(np.arange(0.00, 1.001, 0.02), 2):
            w_swin = 1.0 - w_conv
            blend_log = (w_conv * conv_oof) + (w_swin * swin_oof)
            blend_mae = mean_absolute_error(y_true, calories_from_log(blend_log))
            row = {
                "blend_type": "pair_weighted",
                "convnext_exp": conv_name,
                "swin_exp": swin_name,
                "w_convnext": float(w_conv),
                "w_swin": float(w_swin),
                "oof_mae": float(blend_mae),
            }
            blend_records.append(row)
            if best_blend is None or blend_mae < best_blend["oof_mae"]:
                best_blend = row

if convnext_experiments and swin_experiments:
    common_family_ids = oof_map[convnext_experiments[0]].index
    for name in convnext_experiments[1:] + swin_experiments:
        common_family_ids = common_family_ids.intersection(oof_map[name].index)

    conv_family_oof = np.mean([oof_map[name].loc[common_family_ids].values for name in convnext_experiments], axis=0)
    swin_family_oof = np.mean([oof_map[name].loc[common_family_ids].values for name in swin_experiments], axis=0)
    y_true_family = train_df.set_index("image_id").loc[common_family_ids, gcfg.target_col].values

    for w_conv in np.round(np.arange(0.00, 1.001, 0.02), 2):
        w_swin = 1.0 - w_conv
        family_blend_log = (w_conv * conv_family_oof) + (w_swin * swin_family_oof)
        family_mae = mean_absolute_error(y_true_family, calories_from_log(family_blend_log))
        row = {
            "blend_type": "family_weighted",
            "convnext_exp": "convnext_family_mean",
            "swin_exp": "swin_family_mean",
            "w_convnext": float(w_conv),
            "w_swin": float(w_swin),
            "oof_mae": float(family_mae),
        }
        blend_records.append(row)
        if best_blend is None or family_mae < best_blend["oof_mae"]:
            best_blend = row

blend_df = pd.DataFrame(blend_records).sort_values("oof_mae").reset_index(drop=True)
blend_df.to_csv(Path(gcfg.out_dir) / "blend_search_oof.csv", index=False)
print("Top blend candidates by OOF MAE:")
display(blend_df.head(10))

if best_blend is not None:
    # Export both the best blend and a leaderboard probe set of top-ranked blends.
    if best_blend["blend_type"] == "family_weighted":
        conv_test_mean = np.mean([test_pred_log_map[name] for name in convnext_experiments], axis=0)
        swin_test_mean = np.mean([test_pred_log_map[name] for name in swin_experiments], axis=0)
        best_test_log = (best_blend["w_convnext"] * conv_test_mean) + (best_blend["w_swin"] * swin_test_mean)
    elif best_blend["blend_type"] == "single":
        target_name = best_blend["convnext_exp"] if best_blend["convnext_exp"] else best_blend["swin_exp"]
        best_test_log = test_pred_log_map[target_name]
    else:
        conv_name = best_blend["convnext_exp"]
        swin_name = best_blend["swin_exp"]
        w_conv = best_blend["w_convnext"]
        w_swin = best_blend["w_swin"]
        best_test_log = (w_conv * test_pred_log_map[conv_name]) + (w_swin * test_pred_log_map[swin_name])
    best_blend_submission = pd.DataFrame({
        "image_id": image_ids_reference,
        "predicted_calories": calories_from_log(best_test_log),
    })
    best_blend_submission = test_df[["image_id"]].merge(best_blend_submission, on="image_id", how="left")
    best_blend_submission.to_csv(Path(gcfg.out_dir) / "submission_best_weighted_blend.csv", index=False)

    # Export a few top weighted blend submissions for leaderboard probing.
    for rank, row in blend_df.head(12).iterrows():
        blend_type = row["blend_type"]
        if blend_type == "family_weighted":
            conv_test_mean = np.mean([test_pred_log_map[name] for name in convnext_experiments], axis=0)
            swin_test_mean = np.mean([test_pred_log_map[name] for name in swin_experiments], axis=0)
            blend_test_log = (float(row["w_convnext"]) * conv_test_mean) + (float(row["w_swin"]) * swin_test_mean)
            blend_tag = f"family_conv_w{float(row['w_convnext']):.2f}_swin_w{float(row['w_swin']):.2f}"
        elif blend_type == "single":
            target_name = row["convnext_exp"] if row["convnext_exp"] else row["swin_exp"]
            blend_test_log = test_pred_log_map[target_name]
            blend_tag = f"single_{target_name}"
        else:
            conv_name = row["convnext_exp"]
            swin_name = row["swin_exp"]
            w_conv = float(row["w_convnext"])
            w_swin = float(row["w_swin"])
            blend_test_log = (w_conv * test_pred_log_map[conv_name]) + (w_swin * test_pred_log_map[swin_name])
            blend_tag = f"pair_{conv_name}_w{w_conv:.2f}_{swin_name}_w{w_swin:.2f}"

        blend_submission = pd.DataFrame({
            "image_id": image_ids_reference,
            "predicted_calories": calories_from_log(blend_test_log),
        })
        blend_submission = test_df[["image_id"]].merge(blend_submission, on="image_id", how="left")
        out_name = f"submission_blend_rank{rank + 1}_{blend_tag}.csv"
        blend_submission.to_csv(Path(gcfg.out_dir) / out_name, index=False)

pd.DataFrame(prediction_records).to_csv(Path(gcfg.out_dir) / "prediction_checkpoints.csv", index=False)
display(submission.head())
display(submission["predicted_calories"].describe())
print("Saved run-specific equal-mean submission.")
print(f"Run output directory: {gcfg.out_dir}")
print(f"Saved run-specific equal-mean submission to {run_submission_path}")


# Section: Submission Format Validation


if Path("sample_submission.csv").exists():
    sample = pd.read_csv("sample_submission.csv")
    assert list(submission.columns) == list(sample.columns), (submission.columns.tolist(), sample.columns.tolist())
    assert submission.shape == sample.shape, (submission.shape, sample.shape)
    assert set(submission["image_id"]) == set(sample["image_id"])

assert submission["predicted_calories"].between(gcfg.pred_clip_min, gcfg.pred_clip_max).all()
print(submission.head(10).to_string(index=False))
print("Submission format checks passed.")


# Section: Prediction Distribution Diagnostics


plt.figure(figsize=(8, 4))
sns.histplot(submission["predicted_calories"], bins=60, label="test predictions", color="tab:orange")
sns.histplot(train_df[gcfg.target_col], bins=60, label="train labels", color="tab:blue", alpha=0.35)
plt.legend()
plt.title("Train labels vs test predictions")
plt.tight_layout()
plt.show()


# ## Final Run Notes
# 
# `MODE = "final"` now focuses only on ConvNeXt Large (448) and SwinV2 Large (384), each with multiple tuned variants and longer schedules.