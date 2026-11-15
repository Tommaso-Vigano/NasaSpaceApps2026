import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.server.server import WebServer
from src.landmarks.registry import LandmarkRegistry
from src.data.data_model import WorldModel
from src.data.ingestion import load_datasets

# Get the root of the folder
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:

    # Load the dotenv and parse everything.
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="MarsWalk landmark API")
    parser.add_argument("--landmarks", default=os.getenv("LANDMARKS_PATH"),
                        help="Preset JSON path (defaults to LANDMARKS_PATH in .env)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--datasets-config", default=os.getenv("DATASETS_CONFIG"),
                        help="Dataset configuration JSON (defaults to DATASETS_CONFIG in .env)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--globe-layers", default=os.getenv("GLOBE_LAYERS_PATH", "data/globe_layers"),
                        help="Directory containing global albedo and thermal-inertia TIFFs")
    args = parser.parse_args()

    if not args.landmarks:
        parser.error("The server needs to be able to serve landmarks "
                     "for it to visualize the surface of mars.")
    if not args.datasets_config:
        parser.error("Set DATASETS_CONFIG in .env or pass --datasets-config.")
    try:
        landmarks = LandmarkRegistry.from_file(args.landmarks)
        world_model = WorldModel(load_datasets(args.datasets_config))
    except (OSError, ValueError) as error:
        parser.error(str(error))


    # Begin hosting the webserver
    globe_layers_path = Path(args.globe_layers)
    if not globe_layers_path.is_absolute():
        globe_layers_path = ROOT / globe_layers_path
    WebServer(landmarks, world_model, globe_layers_path).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
