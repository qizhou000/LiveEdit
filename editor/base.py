from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import json, yaml, os


@dataclass
class BaseConfig:
    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
        return cls(**data)
    @classmethod
    def from_yaml(cls, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    def to_dict(config) -> dict:
        dict = asdict(config)
        return dict

