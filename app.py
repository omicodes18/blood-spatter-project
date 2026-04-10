from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import math
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def preprocess_image(gray_image):
    """Keep pipeline simple and similar: blur + edge extraction."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def detect_valid_contours(edges, min_area=100.0):
    """Find external contours and remove tiny noise blobs."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= float(min_area)]
    return valid


def contour_metrics(contour):
    """Compute width/length/angle/confidence for one contour safely."""
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    width = min(w, h)
    length = max(w, h)

    if length <= 0:
        return None

    ratio = max(0.0, min(1.0, width / length))
    angle = math.degrees(math.asin(ratio))
    confidence = ratio * 100.0
    return {
        "rect": rect,
        "width": float(width),
        "length": float(length),
        "angle": float(angle),
        "confidence": float(confidence),
    }


def forensic_conclusion(avg_angle):
    if avg_angle < 30.0:
        return "Shallow impact pattern: droplets indicate low-angle blood travel."
    if avg_angle <= 60.0:
        return "Angled impact pattern: droplets indicate oblique blood travel."
    return "Near-perpendicular impact pattern: droplets indicate steep blood travel."


def analyze_image(image_path, min_area=100.0):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Invalid image data"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = preprocess_image(gray)
    valid_contours = detect_valid_contours(edges, min_area=min_area)

    if len(valid_contours) == 0:
        return None, "No valid contours found"

    output_image = image.copy()
    angles = []
    confidences = []
    droplet_details = []

    for idx, contour in enumerate(valid_contours, start=1):
        metrics = contour_metrics(contour)
        if metrics is None:
            continue

        rect = metrics["rect"]
        angle = metrics["angle"]
        confidence = metrics["confidence"]

        angles.append(angle)
        confidences.append(confidence)
        droplet_details.append(
            {
                "id": idx,
                "angle": round(angle, 2),
                "confidence": round(confidence, 2),
                "width": round(metrics["width"], 2),
                "length": round(metrics["length"], 2),
            }
        )

        # Draw each droplet boundary.
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

        # Label near the droplet center using small font to reduce clutter.
        center_x, center_y = np.intp(rect[0])
        label = f"{angle:.1f} deg"
        cv2.putText(
            output_image,
            label,
            (center_x + 4, center_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if len(angles) == 0:
        return None, "No valid droplet geometries found"

    avg_angle = float(np.mean(angles))
    std_angle = float(np.std(angles))
    confidence_avg = float(np.mean(confidences))
    droplet_count = len(angles)

    cv2.putText(
        output_image,
        f"Droplets: {droplet_count}  Avg angle: {avg_angle:.2f} deg",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    result = {
        "angles": [round(a, 2) for a in angles],
        "average_angle": round(avg_angle, 2),
        "std_angle": round(std_angle, 2),
        "confidence_avg": round(confidence_avg, 2),
        "droplet_count": droplet_count,
        "conclusion": forensic_conclusion(avg_angle),
        "droplets": droplet_details,
        "output_image_data": output_image,
    }
    return result, None

@app.route("/")
def home():
    return "Backend Running", 200
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file in request"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    input_path = os.path.join(UPLOAD_FOLDER, safe_name)
    output_filename = "output_" + safe_name
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    file.save(input_path)

    analysis, error = analyze_image(input_path, min_area=100.0)

    if error:
        return jsonify({"error": error}), 400

    cv2.imwrite(output_path, analysis["output_image_data"])

    return jsonify({
        "angle": analysis["average_angle"],
        "confidence": analysis["confidence_avg"],
        "angles": analysis["angles"],
        "average_angle": analysis["average_angle"],
        "std_angle": analysis["std_angle"],
        "confidence_avg": analysis["confidence_avg"],
        "droplet_count": analysis["droplet_count"],
        "conclusion": analysis["conclusion"],
        "droplets": analysis["droplets"],
        "output_image": output_filename,
    })

@app.route("/image/<filename>")
def get_image(filename):
    if filename == "__healthcheck__":
        return "OK", 200

    file_path = os.path.join(OUTPUT_FOLDER, filename)

    if not os.path.exists(file_path):
        return "File not found", 404

    return send_file(file_path)



if __name__ == "__main__":
    app.run(debug=True)