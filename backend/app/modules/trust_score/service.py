def compute_trust_score(liveness_result, face_match_result, deepfake_result, ocr_result):
    breakdown = {}

    l_score = liveness_result['real_confidence'] * 100 if liveness_result.get('real_confidence') is not None else 0
    breakdown['Liveness'] = {"score": l_score, "weight": 0.25, "reasoning": liveness_result.get('error', f"Confidence: {l_score:.1f}%")}

    m_score = face_match_result['similarity'] * 100 if face_match_result.get('similarity') is not None else 0
    breakdown['Face Match'] = {"score": m_score, "weight": 0.30, "reasoning": face_match_result.get('error', f"Similarity: {m_score:.1f}%")}

    if deepfake_result and 'error' not in deepfake_result:
        d_score = (1 - deepfake_result['fake_confidence']) * 100
        d_reason = f"Fake probability: {deepfake_result['fake_confidence']*100:.1f}%"
    else:
        d_score, d_reason = 100, "No video submitted (N/A)"
    breakdown['Deepfake Check'] = {"score": d_score, "weight": 0.30, "reasoning": d_reason}

    if ocr_result.get('extracted_fields'):
        confs = [item['confidence'] for item in ocr_result.get('raw_items', [])]
        o_score = (sum(confs) / len(confs)) * 100 if confs else 0
        o_reason = f"{len(ocr_result['extracted_fields'])} fields extracted, avg confidence {o_score:.1f}%"
    else:
        o_score, o_reason = 0, "No text fields detected"
    breakdown['Document OCR'] = {"score": o_score, "weight": 0.15, "reasoning": o_reason}

    final = sum(v['score'] * v['weight'] for v in breakdown.values())

    if final >= 80: risk = "HIGH TRUST"
    elif final >= 60: risk = "MEDIUM TRUST"
    elif final >= 40: risk = "LOW TRUST"
    else: risk = "REJECT"

    return {"trust_score": round(final, 2), "risk_level": risk, "breakdown": breakdown}