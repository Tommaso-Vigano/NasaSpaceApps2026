""" Explicit loading and persistence of preset and user landmarks. """

from copy import deepcopy
import json
import math
from pathlib import Path

from src.basic import Coordinate
from src.landmarks.landmark import Landmark


class LandmarkRegistry:

    def __init__(self) -> None:
        self._landmarks: dict[str, Landmark] = {}
        self._user_ids: set[str] = set()
        self._preset_path: Path | None = None

    @classmethod
    def from_file(
        cls, path: str | Path, user_path: str | Path | None = None,
    ) -> "LandmarkRegistry":
        """Load presets, optionally followed by user landmarks.

        A missing user file represents an empty collection. A missing preset
        file is an error. Relative paths resolve against the working directory.
        """
        registry = cls()
        registry._preset_path = Path(path).resolve()
        registry._load(Path(path), user=False)
        if user_path is not None:
            if Path(user_path).resolve() == registry._preset_path:
                raise ValueError("Preset and user files must be different.")
            if Path(user_path).exists():
                registry._load(Path(user_path), user=True)
        return registry

    @staticmethod
    def _parse(landmark_id: str, entry: dict) -> Landmark:
        try:
            if not isinstance(landmark_id, str) or not landmark_id.strip():
                raise ValueError("ID must be a nonempty string")
            if not isinstance(entry, dict):
                raise ValueError("entry must be an object")
            title, description = entry["title"], entry["description"]
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a nonempty string")
            if not isinstance(description, str):
                raise ValueError("description must be a string")

            coordinate = entry["coordinate"]
            lat, lon = coordinate["lat"], coordinate["lon"]

            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in (lat, lon)):
                raise ValueError("latitude and longitude must be finite numbers")
            if not -90 <= lat <= 90:
                raise ValueError("latitude must be between -90 and 90")
            # Preserve the dataset's longitude convention (-180..180 or 0..360).
            extra = entry.get("extra", {})
            if not isinstance(extra, dict):
                raise ValueError("extra must be an object")
            json.dumps(extra, allow_nan=False)
            return Landmark(Coordinate(float(lat), float(lon)), title,
                            description, deepcopy(extra))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid landmark {landmark_id!r}: {error}") from error

    def _load(self, path: Path, *, user: bool) -> None:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key/landmark ID: {key!r}")
                result[key] = value
            return result

        try:
            entries = json.loads(path.read_text(encoding="utf-8"),
                                 object_pairs_hook=unique_object)
            if not isinstance(entries, dict):
                raise ValueError("Expected an object keyed by landmark ID")
            for landmark_id, entry in entries.items():
                if landmark_id in self._landmarks:
                    raise ValueError(f"Duplicate landmark ID: {landmark_id!r}")
                self._landmarks[landmark_id] = self._parse(landmark_id, entry)
                if user:
                    self._user_ids.add(landmark_id)
        except ValueError as error:
            raise ValueError(f"Invalid landmark file {path}: {error}") from error

    def get_available(self) -> list[Landmark]:
        """Return landmarks in loading/insertion order."""
        return list(self._landmarks.values())

    def get(self, landmark_id: str) -> Landmark | None:
        return self._landmarks.get(landmark_id)

    @staticmethod
    def _entry(landmark: Landmark) -> dict:
        return {"title": landmark.title, "description": landmark.description,
                "coordinate": {"lat": landmark.coo.lat, "lon": landmark.coo.lon},
                "extra": landmark.extra}

    def add(self, landmark_id: str, landmark: Landmark) -> None:
        """ Add a validated copy as a user landmark; reject conflicting IDs. """
        self.add_landmarks({landmark_id: self._entry(landmark)})

    def add_landmark(
        self, landmark_id: str, coordinate: Coordinate, title: str,
        description: str = "", extra: dict | None = None,
    ) -> None:
        """ Insert one landmark directly, without constructing a Landmark object. """
        self.add_landmarks({landmark_id: {
            "title": title,
            "description": description,
            "coordinate": {"lat": coordinate.lat, "lon": coordinate.lon},
            "extra": {} if extra is None else extra,
        }})

    def add_landmarks(self, entries: dict[str, dict]) -> None:
        """ Insert a decoded JSON dictionary keyed by landmark ID.

        Entries use the file schema: title, description, coordinate (lat, lon),
        and optional extra. Validate the entire batch before changing the registry.
        Conflicting IDs and invalid entries reject the whole batch. Input data
        is copied, and insertion does not read or write files.
        """
        if not isinstance(entries, dict):
            raise ValueError("Expected a dictionary keyed by landmark ID.")
        additions = {}
        for landmark_id, entry in entries.items():
            if landmark_id in self._landmarks:
                raise ValueError(f"Duplicate landmark ID: {landmark_id!r}")
            additions[landmark_id] = self._parse(landmark_id, entry)
        self._landmarks.update(additions)
        self._user_ids.update(additions)

    def get_in_region(
        self, top_left: Coordinate, bottom_right: Coordinate,
    ) -> list[Landmark]:
        """ Return landmarks in an inclusive rectangle, without longitude wrapping. """
        coordinates = (top_left.lat, top_left.lon, bottom_right.lat, bottom_right.lon)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Region coordinates must be finite.")
        if top_left.lon > bottom_right.lon or top_left.lat < bottom_right.lat:
            raise ValueError("Expected an ordered top-left/bottom-right rectangle.")
        return [landmark for landmark in self._landmarks.values()
                if top_left.lon <= landmark.coo.lon <= bottom_right.lon
                and bottom_right.lat <= landmark.coo.lat <= top_left.lat]


    def save_user_landmarks(self, path: str | Path) -> None:
        """ Save only user landmarks; never overwrite the loaded preset file. """
        path = Path(path)
        if path.resolve() == self._preset_path:
            raise ValueError("Cannot overwrite the preset landmark file.")
        entries = {}
        for landmark_id, landmark in self._landmarks.items():
            if landmark_id in self._user_ids:
                entry = self._entry(landmark)
                self._parse(landmark_id, entry)
                entries[landmark_id] = entry
        content = json.dumps(entries, indent=2, ensure_ascii=False, allow_nan=False)
        path.write_text(content + "\n", encoding="utf-8")
