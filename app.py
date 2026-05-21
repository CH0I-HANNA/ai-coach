import base64
import os
import threading

os.environ["DISPLAY"] = ""
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from core.detector import BowDetector
from core.scene_explainer import SceneExplainer
from core.tts_engine import TTSEngine

app = Flask(__name__)
CORS(app)

detector = BowDetector()
explainer = SceneExplainer()
tts = TTSEngine()

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")


def _decode_frame(b64_string: str):
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    frame_b64 = data.get("frame", "")
    mode = data.get("mode", "crosswalk")

    if not frame_b64:
        return jsonify({"error": "empty frame"}), 400

    frame = _decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "invalid frame"}), 400

    result = detector.detect(frame, mode)

    alert_url = ""
    if result["alert"]:
        tts.cleanup_old_files()
        alert_url = tts.generate(result["alert"])

    return jsonify({
        "detections": result["objects"],
        "alert": result["alert"],
        "alert_type": result["alert_type"],
        "alert_url": alert_url,
        "traffic_light": result["traffic_light"],
    })


@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(force=True)
    frame_b64 = data.get("frame", "")

    frame = _decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "invalid frame"}), 400

    description = explainer.explain(frame)
    tts.cleanup_old_files()
    audio_url = tts.generate(description)

    return jsonify({
        "description": description,
        "audio_url": audio_url,
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    frame_b64 = data.get("frame", "")
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "question is required"}), 400

    frame = _decode_frame(frame_b64)
    if frame is None:
        return jsonify({"error": "invalid frame"}), 400

    answer = explainer.ask(frame, question)
    tts.cleanup_old_files()
    audio_url = tts.generate(answer)

    return jsonify({
        "description": answer,
        "audio_url": audio_url,
    })


@app.route("/audio/<path:filename>")
def audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
