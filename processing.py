import base64
import mimetypes
from io import BytesIO
import json
import os

import numpy as np
from PIL import Image
from openai import OpenAI
from rembg import remove
import mediapipe as mp

# ======= MoWahed logic =======

# 1) OpenAI client + API key

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# 2) Validation prompt
VALIDATION_PROMPT = """
You are an expert validator for Saudi National ID photos. Your role is to analyze the image visually and return validation results ONLY. You are NOT allowed to beautify, compress, resize, enhance, crop, or manipulate the image in any way. You only VALIDATE.

IMPORTANT IDENTITY SAFETY RULE:
- You must NEVER modify the person's face, skin, hair, beard, makeup, eyes, or identity.
- You must NEVER apply compression, blur, sharpening, filters, enhancement, or quality reduction.
- You must NEVER generate or replace any pixels.
- You only analyze and report.

VALIDATION CRITERIA:

1. Face Detection:
- Exactly ONE face must be detected.
- The full face must be visible from forehead to chin.
- Beards are allowed.
- Hijab and headscarves are fully ALLOWED for women.
- Ghutra and shemagh are fully ALLOWED for men.
- FAILURE only if:
  - No face detected
  - More than one face detected
  - Face is heavily occluded

2. Background:
- Background should ideally be pure white.
- If background is light beige, light gray, or off-white → mark as "warning" (auto-fixable).
- If background is dark, colorful, or cluttered → mark as "failed".

3. Lighting:
- Face must be evenly lit.
- No extreme shadows.
- No overexposure.
- If lighting can be corrected safely → "warning".
- If lighting destroys facial visibility → "failed".

4. Head Alignment & Centering:
- The face must be vertically straight (no tilt).
- The face must be centered horizontally.
- The eyes should appear roughly in the upper middle third of the frame.
- Shoulders should appear balanced.
- If alignment is slightly off but mathematically correctable → "warning".
- If the head is clearly tilted or cropped → "failed".

5. Accessories (PROHIBITED ITEMS ONLY):
- Sunglasses → FAILED
- Face masks → FAILED
- Medical masks → FAILED
- Hats and caps → FAILED
- heavy makeup → FAILED
- neck or ear are showing → FAILED
- Regular eyeglasses → FAILED
- Hijab → ALLOWED
- Ghutra & shemagh → ALLOWED

RESPONSE FORMAT (STRICT — JSON ONLY):

{
  "faceDetected": {"status": "passed|failed|warning", "message": "رسالة عربية دقيقة"},
  "background": {"status": "passed|failed|warning", "message": "رسالة عربية دقيقة"},
  "lighting": {"status": "passed|failed|warning", "message": "رسالة عربية دقيقة"},
  "alignment": {"status": "passed|failed|warning", "message": "رسالة عربية دقيقة"},
  "accessories": {"status": "passed|failed|warning", "message": "رسالة عربية دقيقة"},
  "overallScore": 0-100
}

STATUS RULES:
- "passed" = no issue
- "warning" = fixable by system without retaking photo
- "failed" = requires new photo

LANGUAGE RULE:
- ALL messages must be in Arabic.
- Be precise.
- Do not generalize.
- Do not sound robotic.
- Do not mention "AI" or "model".

ONLY return JSON. NOTHING outside JSON.
VERY IMPORTANT:
- Do NOT wrap the JSON in markdown.
- Do NOT add ``` or ```json.
- Return ONLY raw JSON text.
"""

# 3) Helper – encode image as data URL
def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# 4) Helper – extract raw JSON from model text (احتياط لو طلع معه شيء زيادة)
def _extract_raw_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first:last+1]
    return text.strip()

# 5) Call OpenAI validator (vision + JSON)
def validate_id_photo(path: str) -> dict:
    data_url = image_to_data_url(path)

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VALIDATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        },
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
    )

    # في chat.completions نستخدم choices[0].message.content
    raw_text = resp.choices[0].message.content
    cleaned = _extract_raw_json(raw_text)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("=== RAW TEXT FROM MODEL ===")
        print(raw_text)
        print("=== CLEANED TEXT THAT FAILED ===")
        print(cleaned)
        raise ValueError(f"Model did not return valid JSON: {e}")

    return result

# 6) Auto-fix with face detection & proper centering
TARGET_SIZE = (480, 640)  # (width, height)

def _auto_fix_without_face(path: str, out_path: str) -> None:
    img = Image.open(path).convert("RGBA")
    subject = remove(img)

    target_w, target_h = TARGET_SIZE
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))

    subj_w, subj_h = subject.size
    scale = min(target_w / subj_w, target_h / subj_h)
    new_w = int(subj_w * scale)
    new_h = int(subj_h * scale)

    subject_resized = subject.resize((new_w, new_h), Image.LANCZOS)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2

    canvas.paste(subject_resized, (off_x, off_y), subject_resized)
    canvas.convert("RGB").save(out_path, format="JPEG", quality=95)

def auto_fix_image(path: str, out_path: str) -> None:
    orig = Image.open(path).convert("RGB")
    w, h = orig.size

    # 1) كشف الوجه
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        results = fd.process(np.array(orig)[:, :, ::-1])  # BGR

    if not results.detections:
        print(" لم يتم اكتشاف وجه، نستخدم طريقة بديلة بسيطة.")
        _auto_fix_without_face(path, out_path)
        return

    det = results.detections[0]
    bbox = det.location_data.relative_bounding_box
    fx = bbox.xmin * w
    fy = bbox.ymin * h
    fw = bbox.width * w
    fh = bbox.height * h

    face_center_x = fx + fw / 2
    face_center_y = fy + fh / 2

    # 2) إزالة الخلفية
    subject_rgba = remove(orig.convert("RGBA"))
    subj_w, subj_h = subject_rgba.size

    target_w, target_h = TARGET_SIZE

    # نخلي الوجه ياخذ تقريباً نصف ارتفاع الصورة
    desired_face_h = target_h * 0.50
    scale = desired_face_h / fh

    new_subj_w = int(subj_w * scale)
    new_subj_h = int(subj_h * scale)
    subject_scaled = subject_rgba.resize((new_subj_w, new_subj_h), Image.LANCZOS)

    # إعادة حساب مركز الوجه بعد الـ scaling
    scale_x = new_subj_w / w
    scale_y = new_subj_h / h
    face_center_x_scaled = face_center_x * scale_x
    face_center_y_scaled = face_center_y * scale_y
    face_top_y_scaled = fy * scale_y

    # 3) كانفس أبيض بالحجم الرسمي
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))

    # نبي مركز الوجه يكون شوي فوق الوسط، والكتوف قريبة من الأسفل
    desired_face_center_y = target_h * 0.52

    offset_x = int(target_w / 2 - face_center_x_scaled)
    offset_y = int(desired_face_center_y - face_center_y_scaled)

    # تأكد أن أعلى الرأس ما يطلع برا
    head_margin = 10
    top_in_canvas = offset_y + face_top_y_scaled - head_margin
    if top_in_canvas < 0:
        offset_y -= top_in_canvas

    # بعد التمركز الأولي، نخلي أسفل الكتوف قريب من أسفل الصورة
    bottom_in_canvas = offset_y + new_subj_h
    desired_bottom_margin = 8
    extra_space = target_h - bottom_in_canvas
    if extra_space > desired_bottom_margin:
        offset_y += extra_space - desired_bottom_margin

    canvas.paste(subject_scaled, (offset_x, offset_y), subject_scaled)
    final = canvas.convert("RGB")
    final.save(out_path, format="JPEG", quality=95)

# 7) Main function: validate + fix
def process_id_photo(input_path: str, output_path: str):
    print(">> Validating image with OpenAI...")
    result = validate_id_photo(input_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    fatal_keys = ["faceDetected", "alignment", "accessories"]
    has_fatal = any(
        result[key]["status"] == "failed"
        for key in fatal_keys
    )

    if has_fatal:
        print("\n❌ الصورة فيها مشكلة أساسية (وجه / محاذاة / إكسسوارات) – يفضل إعادة التصوير او يتم تعديلها.")
        return

    print("\n>> Auto-fixing image (size + white background + centering)...")
    auto_fix_image(input_path, output_path)
    print(f" Done. Saved fixed image to: {output_path}")


# 8) Run example
if __name__ == "__main__":
    input_path = "img/man.jpg"      # مسار الصورة الأصلية
    output_path = "img/output/output_1.jpg"
    process_id_photo(input_path, output_path)
