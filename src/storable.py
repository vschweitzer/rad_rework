"""
Base class for storing/caching class instances.
"""

import json
import hashlib
import os
from typing import Callable, Any


class Storable:
    def get_dict_representation(self, by_id: bool = True):
        """
        Get dictionary containing information to re-create class instance.
        If by_id is True, other classes that have an id/save function are
        referenced by their ID.
        """
        dict_representation: dict = {}
        return dict_representation

    def to_file(self, file_path: str):
        """
        Save class instance to file
        """
        with open(file_path, "w") as output_file:
            json.dump(self.get_dict_representation(by_id=False), output_file)

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

    def _check_save_dir(self, save_dir: str = "./", create: bool = True):
        if not os.path.exists(save_dir):
            if create:
                os.mkdir(save_dir)
            else:
                raise FileNotFoundError("Save directory does not exist")
        elif not os.path.isdir(save_dir):
            raise FileExistsError("Save path exists but is not a directory")

    def _save_dependencies(self, save_dir: str = "./"):
        """
        Call save method of other classes, reference them by saving their ID.
        """
        pass

    def get_id(
        self, hash_function: Callable = hashlib.sha3_256, buffer_size: int = 4096
    ):
        """
        Consists of the hash of the dictionary representation. May be
        overridden, as long as the ID can reliably identify an object.
        """

        digestor: Any = hash_function()
        digestor.update(json.dumps(self.get_dict_representation()).encode("utf-8"))
        return digestor.hexdigest()

    def save(self, save_dir: str = "./", create: bool = True):
        self._check_save_dir(save_dir=save_dir, create=create)
        self._save_dependencies(save_dir=save_dir)
        save_path: str = os.path.join(
            save_dir, self.__class__.__name__ + "_" + self.get_id() + ".json"
        )
        with open(save_path, "w") as output_file:
            save_data = self.get_dict_representation(by_id=True)
            json.dump(save_data, output_file)
