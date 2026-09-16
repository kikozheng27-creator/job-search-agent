from pathlib import Path

import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config