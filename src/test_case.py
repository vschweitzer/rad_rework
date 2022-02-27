from typing import Any, Callable
import json
import hashlib


import storable


class TestCaseBase(storable.Storable):
    scan_path: str
    anno_path: str
    category: int
    id: str

    def __init__(self, scan_path: str, anno_path: str, category: int) -> None:
        super().__init__()
        self.scan_path = scan_path
        self.anno_path = anno_path
        self.category = category
        self.id = self.get_id()

    def get_id(
        self, hash_function: Callable = hashlib.sha3_256, buffer_size: int = 4096
    ):
        """
        Consists of <scan_hash>_<anno_hash>.
        """
        # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

        hashes = []
        for input_path in [self.scan_path, self.anno_path]:
            digestor: Any = hash_function()
            with open(input_path, "rb") as input_file:
                while True:
                    data: bytes = input_file.read(buffer_size)
                    if not data:
                        break
                    digestor.update(data)
                hashes.append(digestor.hexdigest())

        return "_".join(hashes)

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["scan_path"] = self.scan_path
        dict_representation["anno_path"] = self.anno_path
        dict_representation["category"] = self.category
        dict_representation["id"] = self.id
        return dict_representation

    def __str__(self) -> str:
        return json.dumps(self.get_dict_representation())

    def to_file(self, file_path: str):
        with open(file_path, "w") as output_file:
            json.dump(self.get_dict_representation(), output_file)

    @classmethod
    def from_file(cls, file_path: str):
        dict_representation: dict = {}
        with open(file_path, "r") as input_file:
            dict_representation = json.load(input_file)

        return cls(
            dict_representation["scan_path"],
            dict_representation["anno_path"],
            dict_representation["category"],
        )

    @classmethod
    def from_scan_path(
        cls,
        scan_path: str,
        category: int,
        anno_suffix: str = "A",
        file_ending: str = ".nii.gz",
    ):
        no_ending: str
        if scan_path.endswith(file_ending):
            no_ending = scan_path[: -len(file_ending)]
        else:
            no_ending = scan_path
        anno_path: str = no_ending + anno_suffix + file_ending
        return cls(scan_path, anno_path, category)


class TestCase(TestCaseBase):
    def __init__(self, scan_path: str, anno_path: str, category: int) -> None:
        super().__init__(scan_path, anno_path, category)
        self.cached: dict = {}

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["cached"] = self.cached
        return dict_representation


if __name__ == "__main__":
    test_scan_path: str = "../Dataset_V2/MR100.nii.gz"
    test_save_path: str = "./100_save.txt"
    tc: TestCase = TestCase.from_scan_path(scan_path=test_scan_path, category=0)
    tc.to_file(test_save_path)
    tc2: TestCase = TestCase.from_file(test_save_path)

    print(tc)
    print(tc2)
