"""Minimal landmark JSON API. One registry is shared within a server instance."""

from threading import RLock, BoundedSemaphore
from pathlib import Path
from io import BytesIO
from hashlib import sha256

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import HTTPException

from src.landmarks.registry import LandmarkRegistry
from src.data.data_model import WorldModel
from src.data.types import Datatype
from src.server.globe_layers import GlobeLayers
from src.server.regions import Regions
from src.server.perseverance_photos import PerseverancePhotos


class WebServer:

    def __init__(self, landmarks: LandmarkRegistry, world_model: WorldModel,
                 globe_layers_path: str | Path | None = None,
                 perseverance_photos_path: str | Path | None = None) -> None:

        self.landmarks = landmarks
        self.world_model = world_model
        self.regions = Regions(world_model)
        self.perseverance_photos = PerseverancePhotos(
            self.regions, perseverance_photos_path if perseverance_photos_path is not None
            else Path(__file__).resolve().parents[2] / 'data/perseverance_photos',
        )
        self.globe_layers = GlobeLayers(
            globe_layers_path if globe_layers_path is not None
            else Path(__file__).resolve().parents[2] / "data/globe_layers"
        )
        self._lock = RLock()
        self._routing_slot = BoundedSemaphore(1)
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
        @self.app.get("/")
        def index():
            return render_template("index.html")

        @self.app.get("/planet")
        def planet():
            return render_template("index_planet.html")

        @self.app.get("/region/<sector_id>")
        def region_page(sector_id):
            if sector_id not in self.regions.sectors:
                abort(404, description="Unknown sector.")
            try:
                names = ("west", "south", "east", "north")
                bounds = {name: float(request.args[name]) for name in names} if request.args else None
                clipped = self.regions.select(sector_id, bounds)
                info = self.regions.info(sector_id, clipped)
            except KeyError:
                abort(400, description="Supply west, south, east and north together.")
            except (ValueError, OSError) as error:
                abort(400, description=str(error))
            info["frame_url"] = url_for("region_frame", sector_id=sector_id)
            info["route_url"] = url_for("region_route", sector_id=sector_id)
            if self.perseverance_photos.available(sector_id):
                info['photos_url'] = url_for('region_photos', sector_id=sector_id)
            return render_template("region.html", region=info)

        @self.app.get('/api/regions/<sector_id>/photos')
        def region_photos(sector_id):
            try:
                return jsonify(self.perseverance_photos.collection(sector_id))
            except KeyError:
                abort(404, description='No mission photos are available for this sector.')
            except (ValueError, OSError) as error:
                self.app.logger.warning('Cannot load Perseverance photos: %s', error)
                abort(422, description='Could not load the local photo collection.')

        @self.app.post("/api/regions/<sector_id>/route")
        def region_route(sector_id):
            from src.server.routing_service import route_region, NoRouteError
            from src.app.penalty.penalty_manager import PenaltyError
            if sector_id not in self.regions.sectors:
                abort(404, description="Unknown sector.")
            payload = request.get_json()
            if not self._routing_slot.acquire(blocking=False):
                abort(429, description="Another route is being computed. Please try again shortly.")
            try:
                return jsonify(route_region(self.regions, sector_id, payload))
            except NoRouteError as error:
                return jsonify(error=str(error), heatmaps=error.heatmaps), 422
            except PenaltyError as error:
                abort(422, description=str(error))
            except TimeoutError as error:
                abort(504, description=str(error))
            except (ValueError, OSError) as error:
                abort(400, description=str(error))
            finally:
                self._routing_slot.release()

        @self.app.get("/api/regions/<sector_id>/frame.png")
        def region_frame(sector_id):
            if sector_id not in self.regions.sectors:
                abort(404, description="Unknown sector.")
            try:
                coordinates = [float(request.args[name]) for name in ("x", "y", "width", "height")]
                pixels = int(request.args.get("pixels", 1024))
                png, viewport = self.regions.frame(sector_id, *coordinates, pixels=pixels)
            except KeyError:
                abort(400, description="Supply x, y, width and height for the viewport.")
            except (OSError, ValueError) as error:
                abort(400, description=str(error))
            response = send_file(BytesIO(png), mimetype="image/png", max_age=60,
                                 conditional=True, etag=sha256(png).hexdigest())
            response.headers["X-Viewport"] = self.app.json.dumps(viewport)
            return response

        @self.app.get("/api/globe/layers")
        def globe_layer_catalog():
            layers = self.globe_layers.catalog()
            for layer in layers:
                layer["metadata_url"] = url_for("globe_layer_metadata", layer_id=layer["id"])
            return jsonify({"layers": layers})

        def rendered_layer(layer_id):
            try:
                return self.globe_layers.render(layer_id)
            except KeyError:
                abort(404, description="Unknown globe layer.")
            except FileNotFoundError:
                abort(404, description="This globe layer's source TIFF is not installed.")
            except (OSError, ValueError) as error:
                self.app.logger.warning("Cannot render globe layer %s: %s", layer_id, error)
                abort(422, description="Could not render this georeferenced scalar or RGB(A) TIFF.")

        @self.app.get("/api/globe/layers/<layer_id>")
        def globe_layer_metadata(layer_id):
            _, metadata = rendered_layer(layer_id)
            return jsonify({**metadata, "texture_url": url_for(
                "globe_layer_image", layer_id=layer_id, v=metadata["version"],
            )})

        @self.app.get("/api/globe/layers/<layer_id>/texture.png")
        def globe_layer_image(layer_id):
            png, _ = rendered_layer(layer_id)
            return send_file(BytesIO(png), mimetype="image/png", max_age=3600,
                             conditional=True, etag=sha256(png).hexdigest())

        @self.app.get("/api/globe/landmarks")
        def globe_landmarks():
            with self._lock:
                return jsonify(self.landmarks.to_globe_dict())

        @self.app.get("/api/globe/elevation-coverage")
        def elevation_coverage():
            # The viewer displays the union of these footprints. No raster
            # samples or routing computations are needed for this layer.
            return jsonify({
                "rectangles": [
                    {**({"sector_id": self.regions.id_for_map(map_),
                         "viewer_url": url_for("region_page", sector_id=self.regions.id_for_map(map_))}
                        if self.regions.id_for_map(map_) else {}),
                     "top_left": list(map_.top_left),
                     "bottom_right": list(map_.bottom_right)}
                    for map_ in self.world_model.get_maps_by_datatype(Datatype.ELEVATION)
                ],
                "coordinate_order": "longitude, latitude",
            })

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
