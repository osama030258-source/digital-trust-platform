import os
import tempfile
from fastapi import APIRouter, UploadFile, File
from backend.app.core.model_loader import get_models
from backend.app.modules.face_liveness.service import predict_liveness, face_match
from backend.app.modules.deepfake.service import predict_deepfake
from backend.app.modules.document_intel.service import extract_id_fields
from backend.app.modules.trust_score.service import compute_trust_score

router = APIRouter()


@router.post("/verify/liveness")
async def verify_liveness(file: UploadFile = File(...)):
    models_dict = get_models()

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = predict_liveness(tmp_path, models_dict)
    return result
@router.post("/verify/face-match")
async def verify_face_match(selfie: UploadFile = File(...), document: UploadFile = File(...)):
    models_dict = get_models()

    tmp_dir = tempfile.mkdtemp()
    selfie_path = os.path.join(tmp_dir, "selfie_" + selfie.filename)
    doc_path = os.path.join(tmp_dir, "doc_" + document.filename)

    with open(selfie_path, "wb") as f:
        f.write(await selfie.read())
    with open(doc_path, "wb") as f:
        f.write(await document.read())

    result = face_match(selfie_path, doc_path, models_dict)
    return result
@router.post("/verify/deepfake")
async def verify_deepfake(video: UploadFile = File(...)):
    models_dict = get_models()

    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, video.filename)

    with open(video_path, "wb") as f:
        f.write(await video.read())

    result = predict_deepfake(video_path, models_dict)
    return result
@router.post("/verify/document-ocr")
async def verify_document_ocr(document: UploadFile = File(...)):
    models_dict = get_models()

    tmp_dir = tempfile.mkdtemp()
    doc_path = os.path.join(tmp_dir, document.filename)

    with open(doc_path, "wb") as f:
        f.write(await document.read())

    result = extract_id_fields(doc_path, models_dict)
    return result
@router.post("/trust-score")
async def get_trust_score(
    selfie: UploadFile = File(...),
    document: UploadFile = File(...),
    video: UploadFile = File(None)
):
    models_dict = get_models()
    tmp_dir = tempfile.mkdtemp()

    selfie_path = os.path.join(tmp_dir, "selfie_" + selfie.filename)
    doc_path = os.path.join(tmp_dir, "doc_" + document.filename)

    with open(selfie_path, "wb") as f:
        f.write(await selfie.read())
    with open(doc_path, "wb") as f:
        f.write(await document.read())

    liveness_result = predict_liveness(selfie_path, models_dict)
    face_match_result = face_match(selfie_path, doc_path, models_dict)

    deepfake_result = None
    if video:
        video_path = os.path.join(tmp_dir, video.filename)
        with open(video_path, "wb") as f:
            f.write(await video.read())
        deepfake_result = predict_deepfake(video_path, models_dict)

    ocr_result = extract_id_fields(doc_path, models_dict)

    return compute_trust_score(liveness_result, face_match_result, deepfake_result, ocr_result)