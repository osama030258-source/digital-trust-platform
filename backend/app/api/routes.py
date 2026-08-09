import os
import tempfile
from fastapi import APIRouter, UploadFile, File
from backend.app.core.model_loader import get_models
from backend.app.modules.face_liveness.service import predict_liveness, face_match
from backend.app.modules.deepfake.service import predict_deepfake
from backend.app.modules.document_intel.service import extract_id_fields
from backend.app.modules.trust_score.service import compute_trust_score
from backend.app.modules.copilot.service import ask_copilot, generate_investigation_summary
from backend.app.modules.knowledge_graph.service import (
    add_verification_event,
    detect_shared_document,
    detect_shared_device,
    get_user_network,
    get_graph_stats,
)

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
from pydantic import BaseModel


class VerificationEvent(BaseModel):
    user_id: str
    document_id: str
    device_id: str
    ip_address: str
    trust_score: float
    risk_level: str


@router.post("/graph/event")
def log_verification_event(event: VerificationEvent):
    result = add_verification_event(
        event.user_id, event.document_id, event.device_id,
        event.ip_address, event.trust_score, event.risk_level
    )
    return result


@router.get("/graph/check-document/{document_id}")
def check_shared_document(document_id: str):
    return detect_shared_document(document_id)


@router.get("/graph/check-device/{device_id}")
def check_shared_device(device_id: str):
    return detect_shared_device(device_id)


@router.get("/graph/user/{user_id}")
def get_user_connections(user_id: str):
    return get_user_network(user_id)


@router.get("/graph/stats")
def graph_statistics():
    return get_graph_stats()
class CopilotQuestion(BaseModel):
    question: str
    context_data: dict


class SummaryRequest(BaseModel):
    context_data: dict


@router.post("/copilot/ask")
def copilot_ask(request: CopilotQuestion):
    result = ask_copilot(request.question, request.context_data)
    return result


@router.post("/copilot/summary")
def copilot_summary(request: SummaryRequest):
    result = generate_investigation_summary(request.context_data)
    return result