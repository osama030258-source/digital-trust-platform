import os
import torch
import torch.nn as nn
from torchvision import models
from facenet_pytorch import MTCNN, InceptionResnetV1
import easyocr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AI_MODELS_DIR = os.path.join(BASE_DIR, "ai_models")

_device = None
_mtcnn = None
_resnet = None
_liveness_model = None
_deepfake_model = None
_ocr_reader = None


def load_all_models():
    global _device, _mtcnn, _resnet, _liveness_model, _deepfake_model, _ocr_reader

    if _device is not None:
        return  # already loaded

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _mtcnn = MTCNN(image_size=160, margin=20, device=_device)
    _resnet = InceptionResnetV1(pretrained='vggface2').eval().to(_device)

    _liveness_model = models.resnet18(weights=None)
    _liveness_model.fc = nn.Sequential(
        nn.Linear(_liveness_model.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
    )
    _liveness_model.load_state_dict(torch.load(os.path.join(AI_MODELS_DIR, "liveness_model_best.pth"), map_location=_device))
    _liveness_model = _liveness_model.to(_device).eval()

    _deepfake_model = models.resnet18(weights=None)
    _deepfake_model.fc = nn.Sequential(
        nn.Linear(_deepfake_model.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
    )
    _deepfake_model.load_state_dict(torch.load(os.path.join(AI_MODELS_DIR, "deepfake_model_best.pth"), map_location=_device))
    _deepfake_model = _deepfake_model.to(_device).eval()

    _ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    print("✅ All models loaded successfully")


def get_models():
    if _device is None:
        load_all_models()
    return {
        "device": _device,
        "mtcnn": _mtcnn,
        "resnet": _resnet,
        "liveness_model": _liveness_model,
        "deepfake_model": _deepfake_model,
        "ocr_reader": _ocr_reader,
    }