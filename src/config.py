# src/config.py
import torch

# Các đường dẫn
TEST_DIR = "data/test"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_PATH = "models/dongtruong.pth"

# Tham số thiết bị
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tham số model
NUM_CLASSES = 2
IMAGE_SIZE = 224

# Tham số training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 15