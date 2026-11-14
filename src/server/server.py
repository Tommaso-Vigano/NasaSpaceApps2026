"""Minimal landmark JSON API. One registry is shared within a server instance."""

from threading import RLock

from flask import Flask, abort, jsonify, request, url_for
from werkzeug.exceptions import HTTPException

from src.landmarks.registry import LandmarkRegistry


class WebServer:

    def __init__(self, landmarks: LandmarkRegistry) -> None:

        self.landmarks = landmarks
        self._lock = RLock()
        self.app = Flask(__name__)
        self.app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
        self._register_routes()

    def _insert(self, entries: dict) -> None:
        # Serialize duplicate checks and insertion for concurrent browser requests.
        with self._lock:
            if not isinstance(entries, dict):
                abort(400, description="Expected a JSON object keyed by landmark ID.")
            for landmark_id in entries:
                if self.landmarks.get(landmark_id) is not None:
                    abort(409, description=f"Landmark ID already exists: {landmark_id!r}")
            try:
                self.landmarks.add_landmarks(entries)
            except ValueError as error:
                abort(400, description=str(error))

    def _register_routes(self) -> None:
        @self.app.errorhandler(HTTPException)
        def http_error(error):
            response = error.get_response()
            response.data = self.app.json.dumps({"error": error.description})
            response.content_type = "application/json"
            return response

        @self.app.get("/api/landmarks")
        def list_landmarks():
            with self._lock:
                return jsonify(self.landmarks.to_dict())

        @self.app.get("/api/landmarks/<landmark_id>")
        def get_landmark(landmark_id):
            with self._lock:
                landmark = self.landmarks.get(landmark_id)
                if landmark is None:
                    abort(404, description="Landmark not found.")
                return jsonify(self.landmarks.to_dict()[landmark_id])

        @self.app.post("/api/landmarks/<landmark_id>")
        def add_landmark(landmark_id):
            entry = request.get_json()
            self._insert({landmark_id: entry})
            response = jsonify({"added": [landmark_id]})
            response.status_code = 201
            response.headers["Location"] = url_for("get_landmark", landmark_id=landmark_id)
            return response

        @self.app.post("/api/landmarks")
        def add_landmarks():
            entries = request.get_json()
            self._insert(entries)
            return jsonify({"added": list(entries)}), 201

    def run(self, host: str = "127.0.0.1", port: int = 5000) -> None:
        """ Run Flask's local development server. """
        self.app.run(host=host, port=port, debug=False)
