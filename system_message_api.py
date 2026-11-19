"""Shared Flask route for recording the most recent TTS "system message".

Both avro_stream_poller_with_ml.py and mock_poller.py expose this so the
dashboard can show what the voice system said (and when), as a fallback if
a screen recording's audio capture fails or system TTS audio isn't picked
up. construction_stress_speech.py posts to this endpoint right alongside
every speak() call.
"""
from datetime import datetime

from flask import jsonify, request


def register_system_message_routes(app, latest_sample, lock=None):
    """Adds POST/GET /system-message to a Flask app.

    Writes into latest_sample["system_message"] so it also rides along on
    the existing /latest response -- no second polling loop needed on the
    dashboard side.
    """
    latest_sample.setdefault("system_message", None)

    def _write(entry):
        if lock:
            with lock:
                latest_sample["system_message"] = entry
        else:
            latest_sample["system_message"] = entry

    @app.route("/system-message", methods=["POST"])
    def post_system_message():
        body = request.get_json(force=True, silent=True) or {}
        text = (body.get("text") or "").strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        entry = {
            "text": text,
            "kind": body.get("kind", "info"),
            "said_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write(entry)
        return jsonify({"ok": True})

    @app.route("/system-message", methods=["GET"])
    def get_system_message():
        return jsonify(latest_sample.get("system_message"))
