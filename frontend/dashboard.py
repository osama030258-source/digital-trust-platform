import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1
import easyocr
import cv2
import os, tempfile

st.set_page_config(page_title="Enterprise Identity Verification", layout="wide")

# ==================== PATH SETUP ====================
# frontend/dashboard.py -> go up one level to project root -> ai_models/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_MODELS_DIR = os.path.join(BASE_DIR, "ai_models")

# ==================== MODEL LOADING (cached) ====================

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
    model.load_state_dict(torch.load(os.path.join(AI_MODELS_DIR, "liveness_model_best.pth"), map_location=device))
    model = model.to(device).eval()

    model_df = models.resnet18(weights=None)
    model_df.fc = nn.Sequential(nn.Linear(model_df.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
    model_df.load_state_dict(torch.load(os.path.join(AI_MODELS_DIR, "deepfake_model_best.pth"), map_location=device))
    model_df = model_df.to(device).eval()

    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    return device, mtcnn, resnet, model, model_df, reader

device, mtcnn, resnet, model, model_df, reader = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== CORE FUNCTIONS ====================

def predict_liveness_cropped(image_path):
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return {"is_real": None, "real_confidence": None, "error": "No face detected"}
    x1, y1, x2, y2 = boxes[0]
    face_crop = img.crop((x1, y1, x2, y2))
    img_tensor = transform(face_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1)
    return {"is_real": probs[0][1].item() > probs[0][0].item(), "real_confidence": round(probs[0][1].item(), 4)}

def get_face_embedding(image_path):
    img = ImageOps.exif_transpose(Image.open(image_path).convert('RGB'))
    face = mtcnn(img)
    if face is None:
        return None, "No face detected"
    with torch.no_grad():
        return resnet(face.unsqueeze(0).to(device)), None

def face_match(img1, img2, threshold=0.6):
    emb1, err1 = get_face_embedding(img1)
    emb2, err2 = get_face_embedding(img2)
    if emb1 is None or emb2 is None:
        return {"match": None, "similarity": None, "error": err1 or err2}
    sim = F.cosine_similarity(emb1, emb2).item()
    return {"match": sim >= threshold, "similarity": round(sim, 4)}

def extract_face_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

def predict_deepfake(video_path, num_frames=5):
    faces = extract_face_frames(video_path, num_frames)
    if not faces:
        return {"error": "No faces detected in video"}
    probs = []
    with torch.no_grad():
        for face in faces:
            out = model_df(face.unsqueeze(0).to(device))
            probs.append(torch.softmax(out, dim=1)[0][0].item())
    avg = sum(probs) / len(probs)
    return {"is_fake": avg > 0.5, "fake_confidence": round(avg, 4), "frames_analyzed": len(faces)}

def extract_id_fields(image_path):
    results = reader.readtext(image_path)
    items = [(t.strip(), c, b[0][0], b[0][1]) for (b, t, c) in results]
    items = [it for it in items if not (it[2] < 180 and it[0].upper() == "PHOTO")]
    items.sort(key=lambda it: it[3])
    labels = ["Name", "Father Name", "Date of Birth", "ID Number", "Gender", "Expiry Date"]
    extracted, flags = {}, []
    for i, (text, conf, x, y) in enumerate(items):
        for label in labels:
            lc, tc = label.replace(" ", "").lower(), text.replace(":", "").replace(" ", "").lower()
            if tc == lc or tc.startswith(lc):
                if i + 1 < len(items):
                    val, vconf, _, _ = items[i + 1]
                    extracted[label] = val
                    if vconf < 0.7:
                        flags.append(f"Low confidence on '{label}'")
    return {"extracted_fields": extracted, "warnings": flags, "raw_items": items}

def compute_trust_score(selfie_path, doc_path, video_path=None):
    breakdown = {}

    liveness = predict_liveness_cropped(selfie_path)
    l_score = liveness['real_confidence'] * 100 if liveness.get('real_confidence') is not None else 0
    breakdown['Liveness'] = {"score": l_score, "weight": 0.25, "reasoning": liveness.get('error', f"Confidence: {l_score:.1f}%")}

    match = face_match(selfie_path, doc_path)
    m_score = match['similarity'] * 100 if match['similarity'] is not None else 0
    breakdown['Face Match'] = {"score": m_score, "weight": 0.30, "reasoning": match.get('error', f"Similarity: {m_score:.1f}%")}

    if video_path:
        df = predict_deepfake(video_path)
        d_score = (1 - df['fake_confidence']) * 100 if 'error' not in df else 50
        d_reason = df.get('error', f"Fake probability: {df.get('fake_confidence',0)*100:.1f}%")
    else:
        d_score, d_reason = 100, "No video submitted (N/A)"
    breakdown['Deepfake Check'] = {"score": d_score, "weight": 0.30, "reasoning": d_reason}

    ocr = extract_id_fields(doc_path)
    if ocr['extracted_fields']:
        confs = [c for _, c, _, _ in ocr['raw_items']]
        o_score = (sum(confs) / len(confs)) * 100 if confs else 0
        o_reason = f"{len(ocr['extracted_fields'])} fields extracted, avg confidence {o_score:.1f}%"
    else:
        o_score, o_reason = 0, "No text fields detected"
    breakdown['Document OCR'] = {"score": o_score, "weight": 0.15, "reasoning": o_reason}

    final = sum(v['score'] * v['weight'] for v in breakdown.values())
    if final >= 80: risk = "HIGH TRUST"
    elif final >= 60: risk = "MEDIUM TRUST"
    elif final >= 40: risk = "LOW TRUST"
    else: risk = "REJECT"

    return round(final, 2), risk, breakdown

# ==================== UI ====================

st.title("🛡️ Enterprise Identity Verification Platform")
st.caption("AI-powered KYC verification — Liveness · Face Match · Deepfake Detection · Document OCR")

col1, col2, col3 = st.columns(3)
with col1:
    selfie_upload = st.file_uploader("📷 Upload Selfie", type=["jpg", "jpeg", "png"])
with col2:
    doc_upload = st.file_uploader("🪪 Upload ID Document", type=["jpg", "jpeg", "png"])
with col3:
    video_upload = st.file_uploader("🎥 Upload Video (optional)", type=["mp4"])

if st.button("Run Verification", type="primary"):
    if not selfie_upload or not doc_upload:
        st.error("Please upload both a selfie and a document.")
    else:
        with st.spinner("Running verification pipeline..."):
            tmp_dir = tempfile.mkdtemp()
            selfie_path = os.path.join(tmp_dir, "selfie.jpg")
            doc_path = os.path.join(tmp_dir, "doc.jpg")
            with open(selfie_path, "wb") as f: f.write(selfie_upload.read())
            with open(doc_path, "wb") as f: f.write(doc_upload.read())

            video_path = None
            if video_upload:
                video_path = os.path.join(tmp_dir, "video.mp4")
                with open(video_path, "wb") as f: f.write(video_upload.read())

            score, risk, breakdown = compute_trust_score(selfie_path, doc_path, video_path)

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Trust Score", f"{score}/100")
            color = {"HIGH TRUST": "🟢", "MEDIUM TRUST": "🟡", "LOW TRUST": "🟠", "REJECT": "🔴"}
            st.subheader(f"{color.get(risk,'')} {risk}")
        with c2:
            st.progress(min(int(score), 100) / 100)

        st.subheader("Explainability Breakdown")
        for name, data in breakdown.items():
            st.write(f"**{name}** — Score: {data['score']:.1f} (Weight: {data['weight']*100:.0f}%)")
            st.caption(data['reasoning'])
            st.progress(min(int(data['score']), 100) / 100)

        with st.expander("View uploaded images"):
            ic1, ic2 = st.columns(2)
            ic1.image(selfie_path, caption="Selfie", width=250)
            ic2.image(doc_path, caption="Document", width=250)