"""
Base class for storing/caching class instances.
"""

import json


class Storable:
    def get_dict_representation(self):
        """
        Get dictionary containing information to re-create class instance
        """
        dict_representation: dict = {}
        return dict_representation

    def to_file(self, file_path: str):
        """
        Save class instance to file
        """
        with open(file_path, "w") as output_file:
            json.dump(self.get_dict_representation(), output_file)

    @classmethod
    def from_dict(cls, dict_representation: dict):
        """
        Load class from dictionary representation
        """
        return cls()

    @classmethod
    def from_file(cls, file_path: str):
        """
        Load class instance from file
        """
        with open(file_path, "r") as input_file:
            dict_representation = json.load(input_file)
            return cls.from_dict(dict_representation)

    def __str__(self) -> str:
        return json.dumps(self.get_dict_representation())
