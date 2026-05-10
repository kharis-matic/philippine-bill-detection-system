import os
import time
from pathlib import Path
from threading import Lock

import cv2
from flask import Flask, Response, jsonify, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "my_model.pt"
DOWNLOADS_MODEL_PATH = Path.home() / "Downloads" / "my_model.pt"

CONFIDENCE_THRESHOLD = 0.5
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv"}
TARGET_CLASSES = {"new_20peso_bill", "new_50peso_bill"}
BILL_VALUES = {"new_20peso_bill": 20, "new_50peso_bill": 50}
BBOX_COLORS = {
    "new_20peso_bill": (32, 130, 224),
    "new_50peso_bill": (20, 184, 166),
}


UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024


def resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    if DOWNLOADS_MODEL_PATH.exists():
        return DOWNLOADS_MODEL_PATH
    raise FileNotFoundError(
        "Model file not found. Place my_model.pt beside app.py or in your Downloads folder."
    )


model = YOLO(str(resolve_model_path()), task="detect")
class_names = model.names
camera = None
camera_lock = Lock()


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def empty_summary() -> dict:
    return {
        "count_20": 0,
        "count_50": 0,
        "total_value": 0,
        "detections": [],
        "message": "No bill detected.",
    }


def detect_and_annotate(frame, confidence_threshold: float = CONFIDENCE_THRESHOLD):
    results = model(frame, verbose=False, conf=confidence_threshold)
    detections = results[0].boxes
    summary = empty_summary()
    summary["message"] = ""

    for box in detections:
        class_index = int(box.cls.item())
        class_name = class_names[class_index]
        confidence = float(box.conf.item())

        if class_name not in TARGET_CLASSES or confidence < confidence_threshold:
            continue

        xmin, ymin, xmax, ymax = box.xyxy.cpu().numpy().squeeze().astype(int)
        color = BBOX_COLORS.get(class_name, (255, 178, 50))
        denomination = BILL_VALUES[class_name]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{class_name}: {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(ymin, label_size[1] + 12)
        cv2.rectangle(
            frame,
            (xmin, label_y - label_size[1] - 12),
            (xmin + label_size[0] + 8, label_y + baseline - 8),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (xmin + 4, label_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if class_name == "new_20peso_bill":
            summary["count_20"] += 1
        elif class_name == "new_50peso_bill":
            summary["count_50"] += 1

        summary["detections"].append(
            {
                "class_name": class_name,
                "denomination": denomination,
                "confidence": round(confidence, 3),
                "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
            }
        )

    summary["total_value"] = (summary["count_20"] * 20) + (summary["count_50"] * 50)
    if not summary["detections"]:
        summary["message"] = "No 20-peso or 50-peso bill detected."

    overlay = [
        f"20 Peso Bills: {summary['count_20']}",
        f"50 Peso Bills: {summary['count_50']}",
        f"Total: PHP {summary['total_value']}",
    ]
    for idx, text in enumerate(overlay):
        cv2.putText(
            frame,
            text,
            (14, 32 + (idx * 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (18, 94, 176),
            3,
        )
        cv2.putText(
            frame,
            text,
            (14, 32 + (idx * 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
        )

    return frame, summary


def save_upload(file_storage, allowed_extensions: set[str]):
    if not file_storage or file_storage.filename == "":
        return None, ("No file uploaded.", 400)

    if not allowed_file(file_storage.filename, allowed_extensions):
        return None, ("Unsupported file type.", 400)

    timestamp = int(time.time() * 1000)
    filename = secure_filename(file_storage.filename)
    save_path = UPLOAD_FOLDER / f"{timestamp}_{filename}"
    file_storage.save(save_path)
    return save_path, None


def aggregate_summaries(current: dict, frame_summary: dict):
    current["count_20"] += frame_summary["count_20"]
    current["count_50"] += frame_summary["count_50"]
    current["total_value"] += frame_summary["total_value"]
    current["frames_with_detections"] += int(bool(frame_summary["detections"]))
    current["detections"].extend(frame_summary["detections"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect_image", methods=["POST"])
def detect_image():
    image_path, error = save_upload(request.files.get("image"), ALLOWED_IMAGE_EXTENSIONS)
    if error:
        message, status = error
        return jsonify({"error": message}), status

    frame = cv2.imread(str(image_path))
    if frame is None:
        return jsonify({"error": "Could not read the uploaded image."}), 400

    processed_frame, summary = detect_and_annotate(frame)
    result_filename = f"detected_{image_path.stem}.jpg"
    result_path = RESULT_FOLDER / result_filename
    cv2.imwrite(str(result_path), processed_frame)

    return jsonify(
        {
            **summary,
            "image_url": url_for("static", filename=f"results/{result_filename}"),
        }
    )


@app.route("/detect_video", methods=["POST"])
def detect_video():
    video_path, error = save_upload(request.files.get("video"), ALLOWED_VIDEO_EXTENSIONS)
    if error:
        message, status = error
        return jsonify({"error": message}), status

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({"error": "Could not open the uploaded video."}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result_filename = f"processed_{video_path.stem}.mp4"
    result_path = RESULT_FOLDER / result_filename
    writer = cv2.VideoWriter(
        str(result_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    summary = {
        "count_20": 0,
        "count_50": 0,
        "total_value": 0,
        "frames_with_detections": 0,
        "detections": [],
        "message": "",
    }
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        processed_frame, frame_summary = detect_and_annotate(frame)
        writer.write(processed_frame)
        aggregate_summaries(summary, frame_summary)
        frame_count += 1

    cap.release()
    writer.release()

    summary["frame_count"] = frame_count
    if summary["frames_with_detections"] == 0:
        summary["message"] = "No 20-peso or 50-peso bill detected in the video."

    return jsonify(
        {
            **summary,
            "video_url": url_for("static", filename=f"results/{result_filename}"),
        }
    )


def webcam_frames():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)

    while True:
        with camera_lock:
            if camera is None:
                break
            success, frame = camera.read()

        if not success:
            break

        processed_frame, summary = detect_and_annotate(frame)
        status = (
            f"20: {summary['count_20']} | 50: {summary['count_50']} | "
            f"Total: PHP {summary['total_value']}"
        )
        cv2.putText(
            processed_frame,
            status,
            (14, processed_frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        ok, buffer = cv2.imencode(".jpg", processed_frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(webcam_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_webcam", methods=["POST"])
def stop_webcam():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"message": "Webcam stopped."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
