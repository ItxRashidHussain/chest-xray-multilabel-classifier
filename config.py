"""Central configuration for the Chest X-ray multi-label classifier.

Everything you might want to tweak (paths, the 14 disease labels, the backbone,
and training hyper-parameters) lives here so the other scripts stay clean.
"""
from pathlib import Path

# ---------------------------------------------------------------- Paths
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"  # generated: splits, cached features, model, metrics
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _resolve_data_dir():
    """Where the dataset lives.

    If you used download_data.py (kagglehub), it saved the cache path to
    artifacts/data_path.txt and we read it here. Otherwise we fall back to the
    local ./data folder (manual zip download).
    """
    pointer = ARTIFACTS_DIR / "data_path.txt"
    if pointer.exists():
        p = Path(pointer.read_text().strip())
        if p.exists():
            return p
    return ROOT / "data"


DATA_DIR = _resolve_data_dir()  # auto: kagglehub cache if present, else ./data

SPLITS_CSV = ARTIFACTS_DIR / "splits.csv"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODEL_PATH = ARTIFACTS_DIR / "chest_xray_head.pt"
METRICS_PATH = ARTIFACTS_DIR / "metrics.txt"

# ---------------------------------------------------------------- Labels
# The 14 NIH chest-xray findings. This is a MULTI-LABEL problem: one image can
# have several of these at once. "No Finding" is simply an all-zero vector.
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(DISEASE_LABELS)

# ---------------------------------------------------------------- Image preprocessing
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- Backbone
# Frozen, pretrained CNN used only as a feature extractor.
#   "mobilenet_v2" -> 1280-d features, fast (recommended for CPU)
#   "densenet121"  -> 1024-d features, the classic CheXNet backbone (~3x slower on CPU)
BACKBONE = "mobilenet_v2"

# ---------------------------------------------------------------- Feature extraction
EXTRACT_BATCH_SIZE = 32

# ---------------------------------------------------------------- Classifier head training
HIDDEN_DIM = 512
DROPOUT = 0.3
EPOCHS = 30
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 128
WEIGHT_DECAY = 1e-5

# Probability above which a condition is reported as "present" in the demo.
DECISION_THRESHOLD = 0.5

# ---------------------------------------------------------------- Split
# We split by PATIENT so the same person never leaks across train/val/test.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42
