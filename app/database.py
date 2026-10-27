import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_json(filename:str):
    file_path = DATA_DIR / filename

    with file_path.open("r",encoding = "utf-8") as file:
        return json.load(file)

def load_faqs():
    return load_json("faqs.json")

def load_orders():
    return load_json("orders.json")