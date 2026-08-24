from pathlib import Path

from sapai.sim.catalog import Catalog

DATA_PATH = Path(__file__).resolve().parents[1] / "assets" / "data"


def catalog() -> Catalog:
    return Catalog.from_json_dir(DATA_PATH)
