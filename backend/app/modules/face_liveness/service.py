from PIL import Image, ImageOps
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_liveness(image_path, models_dict):
    device = models_dict["device"]
    mtcnn = models_dict["mtcnn"]
    model = models_dict["liveness_model"]

    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        return {"is_real": None, "real_confidence": None, "error": "No face detected"}

    x1, y1, x2, y2 = boxes[0]
    face_crop = img.crop((x1, y1, x2, y2))
    img_tensor = transform(face_crop).unsqueeze(0).to(device)

    import torch
    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1)

    return {
        "is_real": bool(probs[0][1].item() > probs[0][0].item()),
        "real_confidence": round(probs[0][1].item(), 4)
    }
import torch.nn.functional as F


def get_face_embedding(image_path, models_dict):
    device = models_dict["device"]
    mtcnn = models_dict["mtcnn"]
    resnet = models_dict["resnet"]

    img = ImageOps.exif_transpose(Image.open(image_path).convert('RGB'))
    face = mtcnn(img)

    if face is None:
        return None, "No face detected"

    import torch
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))

    return embedding, None


def face_match(img1_path, img2_path, models_dict, threshold=0.6):
    emb1, err1 = get_face_embedding(img1_path, models_dict)
    emb2, err2 = get_face_embedding(img2_path, models_dict)

    if emb1 is None or emb2 is None:
        return {"match": None, "similarity": None, "error": err1 or err2}

    sim = F.cosine_similarity(emb1, emb2).item()

    return {"match": bool(sim >= threshold), "similarity": round(sim, 4)}