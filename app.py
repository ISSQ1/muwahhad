# app.py
import os
import base64
from flask import Flask, request, jsonify, render_template
from processing import validate_id_photo, process_id_photo  # نستخدم دوالك كما هي

UPLOAD_DIR = "img"
OUTPUT_DIR = "img/output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, template_folder=".")

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_route():
    if "photo" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    # حفظ الصورة الأصلية
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    file.save(input_path)

    # مسار الصورة الناتجة
    base_name, _ = os.path.splitext(file.filename)
    output_filename = f"output_{base_name}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1) الفحص الأولي قبل أي معالجة
    validation = validate_id_photo(input_path)

    fatal_keys = ["faceDetected", "alignment", "accessories"]
    fatal_issues = []
    warnings = []

    for key, value in validation.items():
        if key == "overallScore":
            continue
        status = value.get("status")
        msg = value.get("message", "")
        item = {"key": key, "status": status, "message": msg}
        if status == "failed" and key in fatal_keys:
            fatal_issues.append(item)
        elif status == "warning":
            warnings.append(item)

    is_rejected = len(fatal_issues) > 0
    decision = "rejected" if is_rejected else "processed"

    # تجهيز الصورة الأصلية كـ base64 عشان نعرضها حتى لو مرفوضة
    input_b64 = img_to_base64(input_path)

    # لو مرفوض → لا نعمل processing، ونرجّع السبب فقط
    if is_rejected:
        return jsonify({
            "success": True,
            "decision": decision,
            "accepted": False,
            "fatal_issues": fatal_issues,
            "warnings": warnings,
            "validation_before": validation,
            "processing_applied": [],
            "input_image": input_b64,
            "output_image": None
        })

    # 2) مقبول مع أو بدون تحذيرات → نعمل processing
    process_id_photo(input_path, output_path)

    if not os.path.exists(output_path):
        # لو لأي سبب ما طلعت صورة بعد المعالجة
        return jsonify({
            "success": False,
            "decision": "processing_failed",
            "error": "Processing did not produce an output image.",
            "validation_before": validation,
            "input_image": input_b64,
            "output_image": None
        }), 500

    output_b64 = img_to_base64(output_path)

    # وصف عام لما تم تطبيقه في مرحلة المعالجة
    processing_applied = [
        "تم ضبط أبعاد الصورة إلى 480×640 بكسل بما يناسب متطلبات صورة الهوية.",
        "تم توسيط الوجه داخل إطار الصورة بناءً على موقع الوجه المكتشف.",
        "تم إنزال الكتفين بحيث تكون وضعية الجسم متناسقة داخل الإطار.",
        "تم تعيين الخلفية إلى لون أبيض موحد مع الحفاظ على ملامح الوجه كما هي."
    ]

    return jsonify({
        "success": True,
        "decision": decision,
        "accepted": True,
        "fatal_issues": fatal_issues,
        "warnings": warnings,
        "validation_before": validation,
        "processing_applied": processing_applied,
        "input_image": input_b64,
        "output_image": output_b64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
