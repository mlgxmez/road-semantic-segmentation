import json
from typing import Any, Dict


class Augmentation(object):
    """
    Image augmentation is read from a JSON file.
    """
    def __init__(self, file: str) -> None:
        """
        Read Json and store configuration in a dict
        """
        self.file = file
        self.config = self.read_file()

    def read_file(self) -> Dict[str, Any]:
        with open(self.file) as f:
            config = json.load(f)
        return config
