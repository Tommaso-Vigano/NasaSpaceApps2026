import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.server.server import WebServer
from src.landmarks.registry import LandmarkRegistry

# Get the root of the folder
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:

    # Load the dotenv and parse everything.
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="MarsWalk landmark API")
    parser.add_argument("--landmarks", default=os.getenv("LANDMARKS_PATH"),
                        help="Preset JSON path (defaults to LANDMARKS_PATH in .env)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if not args.landmarks:
        parser.error("The server needs to be able to serve landmarks "
                     "for it to visualize the surface of mars.")
    try:
        landmarks = LandmarkRegistry.from_file(args.landmarks)
    except (OSError, ValueError) as error:
        parser.error(str(error))


    # Begin hosting the webserver
    WebServer(landmarks).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
