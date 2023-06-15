import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/train/"
VAL_DIR = "dataset/val/"
LEARNING_RATE = 0.005
BATCH_SIZE = 128
NUM_WORKERS = 2
IMAGE_SIZE = 48
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT = "modelcheckpoint/my_checkpoint.pth.tar"

transformation = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])