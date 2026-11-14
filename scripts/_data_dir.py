"""Choice of the Jezero data folder, shared by the scripts.

Order (provisional, to be decided with the team):
  1. --data-dir, if given;
  2. the environment variable JEZERO_DATA_DIR (also read from REPO_ROOT/.env, which is
     git-ignored, with python-dotenv);
  3. DEFAULT_JEZERO_DATA_DIR: a "jezero_data" folder next to the repository.
If the chosen folder does not exist the script stops with a message that explains the
three options.
"""
import os
from pathlib import Path
from typing import Optional

from src.constants import DEFAULT_JEZERO_DATA_DIR, JEZERO_DATA_DIR_ENV, REPO_ROOT


def resolve_data_dir(cli_value: Optional[Path] = None) -> Path:
    """Data folder from --data-dir, JEZERO_DATA_DIR (environment or .env), or the
    default; SystemExit with an explanation if it does not exist."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv missing: only the real environment is used
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")  # does not override variables already set

    env_value = os.environ.get(JEZERO_DATA_DIR_ENV, "").strip()
    if cli_value is not None:
        chosen, origin = Path(cli_value), "--data-dir"
    elif env_value:
        chosen, origin = Path(env_value), f"variabile {JEZERO_DATA_DIR_ENV}"
    else:
        chosen, origin = DEFAULT_JEZERO_DATA_DIR, "cartella di default"
    chosen = chosen.expanduser()

    if not chosen.is_dir():
        raise SystemExit(
            f"Cartella dati Jezero non trovata: {chosen} (da {origin}).\n"
            "Indicala in uno di questi modi:\n"
            "  1. --data-dir /percorso/jezero_data\n"
            f"  2. {JEZERO_DATA_DIR_ENV}=/percorso/jezero_data nell'ambiente, oppure nel file "
            f"{REPO_ROOT / '.env'} (ignorato da git), tra virgolette se ci sono spazi:\n"
            f'     {JEZERO_DATA_DIR_ENV}="/percorso con spazi/jezero_data"\n'
            f"  3. metti la cartella in {DEFAULT_JEZERO_DATA_DIR} (accanto al repository)."
        )
    return chosen
