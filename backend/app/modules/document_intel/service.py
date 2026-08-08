def extract_id_fields(image_path, models_dict):
    reader = models_dict["ocr_reader"]

    results = reader.readtext(image_path)
    items = [(t.strip(), c, b[0][0], b[0][1]) for (b, t, c) in results]
    items = [it for it in items if not (it[2] < 180 and it[0].upper() == "PHOTO")]
    items.sort(key=lambda it: it[3])

    labels = ["Name", "Father Name", "Date of Birth", "ID Number", "Gender", "Expiry Date"]
    extracted, flags = {}, []

    for i, (text, conf, x, y) in enumerate(items):
        for label in labels:
            lc = label.replace(" ", "").lower()
            tc = text.replace(":", "").replace(" ", "").lower()
            if tc == lc or tc.startswith(lc):
                if i + 1 < len(items):
                    val, vconf, _, _ = items[i + 1]
                    extracted[label] = val
                    if vconf < 0.7:
                        flags.append(f"Low confidence on '{label}'")

    return {
        "extracted_fields": extracted,
        "warnings": flags,
        "raw_items": [{"text": t, "confidence": round(c, 4)} for t, c, _, _ in items]
    }