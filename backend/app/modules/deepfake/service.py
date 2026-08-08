import cv2
from PIL import Image
import torch


def extract_face_frames(video_path, mtcnn, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    idxs = [int(i * total / num_frames) for i in range(num_frames)]
    faces = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(Image.fromarray(frame))
            if face is not None:
                faces.append(face)

    cap.release()
    return faces


def predict_deepfake(video_path, models_dict, num_frames=5):
    device = models_dict["device"]
    mtcnn = models_dict["mtcnn"]
    model_df = models_dict["deepfake_model"]

    faces = extract_face_frames(video_path, mtcnn, num_frames)

    if not faces:
        return {"error": "No faces detected in video"}

    probs = []
    with torch.no_grad():
        for face in faces:
            out = model_df(face.unsqueeze(0).to(device))
            probs.append(torch.softmax(out, dim=1)[0][0].item())

    avg = sum(probs) / len(probs)

    return {
        "is_fake": bool(avg > 0.5),
        "fake_confidence": round(avg, 4),
        "frames_analyzed": len(faces)
    }