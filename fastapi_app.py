"""
FastAPI implementation of the Saudi ID photo validation and processing API.

This module mirrors the behaviour of the original Flask implementation (see
``app.py``) but exposes the API using FastAPI.  The API accepts a JSON
payload containing a base64‐encoded image and returns a JSON response
containing the validation results, any processing applied, and base64
encoded versions of the input and output images.  A permissive CORS
configuration is added so that the React frontend running on a different
origin (e.g. ``npm run dev`` served on ``localhost:3000``) can call the
backend without encountering cross‑origin errors.

How to use:

1. Ensure the dependencies listed in ``requirements.txt`` are installed.
2. Start the backend with:

   ``uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000``

3. Start the React frontend in another terminal as usual (``npm run dev``).
   The frontend has been updated to send a JSON body to the ``/process``
   endpoint on ``localhost:8000``.  When both servers are running you can
   open the frontend in your browser and upload an image for validation
   and processing.
"""

import os
import base64
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from processing import validate_id_photo, process_id_photo


# Directory locations for storing uploaded and processed images.  These
# directories are created at startup if they don't already exist.
UPLOAD_DIR = "img"
OUTPUT_DIR = os.path.join(UPLOAD_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ProcessRequest(BaseModel):
    """Schema for the POST /process request body.

    The frontend sends a JSON object containing a base64 string
    representing the image data and an optional filename.  If no
    filename is provided, a default name will be used.
    """

    image_base64: str
    filename: Optional[str] = None


def image_to_base64(path: str) -> str:
    """Read an image from ``path`` and return a base64 encoded string.

    This helper mirrors the behaviour of the Flask implementation, but
    returns raw base64 data without any MIME prefix so that the
    JavaScript frontend can easily embed it into a data URI.
    """

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_base64_image(data: str) -> bytes:
    """Decode a base64 string into raw bytes.

    Raises ``HTTPException`` with status 400 if the data cannot be
    decoded.
    """

    try:
        return base64.b64decode(data)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64 image data")


def prepare_output_filename(filename: str) -> str:
    """Generate an output filename based on the original filename.

    Given a filename like ``example.jpg`` this function returns
    ``output_example.jpg``.  If no extension is present, ``.jpg`` is
    appended by default.
    """

    base_name, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jpg"
    return f"output_{base_name}{ext}"


app = FastAPI(title="Saudi ID Photo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production you should restrict this list
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    """Basic health check endpoint.

    Returns a simple JSON message indicating that the API is running.
    """

    return {"message": "Saudi ID Photo API is running"}


@app.post("/process")
async def process_endpoint(req: ProcessRequest) -> dict:
    """Validate and optionally process an ID photo.

    This endpoint accepts a base64 encoded image via JSON, saves it to disk
    under ``UPLOAD_DIR``, runs the validation logic from ``processing.py``,
    and if the photo is acceptable, runs the processing logic to centre
    the subject and adjust the background.  The response mirrors the
    structure of the Flask implementation: it includes whether the
    request was successful, the decision (``rejected`` or ``processed``),
    any fatal issues or warnings, the full validation data, a list of
    descriptions of the processing steps applied, and base64 encoded
    versions of the input and output images.
    """

    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # Decode the incoming base64 image
    image_bytes = decode_base64_image(req.image_base64)

    # Determine a filename for saving; default to a generic name
    filename = req.filename or "uploaded_image.jpg"
    input_path = os.path.join(UPLOAD_DIR, filename)

    # Save the uploaded image to disk so that the processing functions
    # operate on a file path as expected
    with open(input_path, "wb") as f:
        f.write(image_bytes)

    # Determine the output path for the processed image
    output_filename = prepare_output_filename(filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Run the initial validation
    validation = validate_id_photo(input_path)

    # Extract fatal issues and warnings from the validation result
    fatal_keys = ["faceDetected", "alignment", "accessories"]
    fatal_issues: List[dict] = []
    warnings: List[dict] = []

    for key, value in validation.items():
        # Skip overallScore as it's a numeric aggregate
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

    # Convert the input image to base64 for the response
    input_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # If the image is rejected, return early without processing
    if is_rejected:
        return {
            "success": True,
            "decision": decision,
            "accepted": False,
            "fatal_issues": fatal_issues,
            "warnings": warnings,
            "validation_before": validation,
            "processing_applied": [],
            "input_image": input_b64,
            "output_image": None,
        }

    # Image is acceptable; perform processing to centre and clean it
    process_id_photo(input_path, output_path)

    if not os.path.exists(output_path):
        # The processing failed to produce an output file; report an error
        return {
            "success": False,
            "decision": "processing_failed",
            "error": "Processing did not produce an output image.",
            "validation_before": validation,
            "input_image": input_b64,
            "output_image": None,
        }

    # Read the processed image into base64
    output_b64 = image_to_base64(output_path)

    # Describe the transformations applied during processing.  These
    # descriptions mirror the ones used in the Flask implementation and
    # should be updated if the processing logic changes.
    processing_applied = [
        "تم ضبط أبعاد الصورة إلى 480×640 بكسل بما يناسب متطلبات صورة الهوية.",
        "تم توسيط الوجه داخل إطار الصورة بناءً على موقع الوجه المكتشف.",
        "تم إنزال الكتفين بحيث تكون وضعية الجسم متناسقة داخل الإطار.",
        "تم تعيين الخلفية إلى لون أبيض موحد مع الحفاظ على ملامح الوجه كما هي."
    ]

    return {
        "success": True,
        "decision": decision,
        "accepted": True,
        "fatal_issues": fatal_issues,
        "warnings": warnings,
        "validation_before": validation,
        "processing_applied": processing_applied,
        "input_image": input_b64,
        "output_image": output_b64,
    }