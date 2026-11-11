# ============================================================
# config.py — subliminal_weight_diff (JUWELS)
# Per-example gradient projection onto the teacher-vs-control
# student LoRA weight diff (v_transfer).
# Supports both "numbers" and "sentences" carrier datasets.
# ============================================================

import os

SOURCE_DIRS = {
    "numbers":   "/p/scratch/hai_1220/kumar32/subliminal_finetuning",
    "sentences": "/p/scratch/hai_1220/kumar32/subliminal_sentences",
}
DATA_SUBDIR = {
    "numbers":   "numbers",
    "sentences": "sentences",
}
BASE_DIR = "/p/scratch/hai_1220/kumar32/subliminal_weight_diff"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

LORA_R              = 16
LORA_ALPHA           = 16
LORA_DROPOUT         = 0.0
LORA_BIAS            = "none"
LORA_USE_RSLORA      = False
LORA_TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

TRAIN_MAX_SEQ_LEN   = 128
TRAIN_SEED          = 1337
TRAIN_LR            = 2e-4
TRAIN_WEIGHT_DECAY  = 0.01

# which trained student epoch to diff (final epoch)
STUDENT_EPOCH = "epoch_10"

def data_path(animal: str, mode: str, dataset: str = "numbers") -> str:
    source_dir = SOURCE_DIRS[dataset]
    subdir     = DATA_SUBDIR[dataset]
    return os.path.join(source_dir, "results", animal, subdir, f"{mode}_filtered.jsonl")

def student_ckpt_path(animal: str, mode: str, dataset: str = "numbers") -> str:
    source_dir = SOURCE_DIRS[dataset]
    return os.path.join(source_dir, "results", animal, "student_checkpoints", mode, STUDENT_EPOCH)

def results_dir(animal: str, dataset: str = "numbers") -> str:
    d = os.path.join(BASE_DIR, "results", dataset, animal)
    os.makedirs(d, exist_ok=True)
    return d

def make_dirs():
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
