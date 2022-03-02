""" 
Handles the representation of patients/test cases
"""

from typing import Any, Callable, Optional
import json
import hashlib
import nibabel
import numpy as np

import storable


class TestCaseBase(storable.Storable):
    """
    Represents a scan/annotation pair with health information.
    """

    # stored
    scan_path: str
    anno_path: str
    category: int
    id: str

    # not stored
    anno_object: Any
    scan_object: Any

    def __init__(self, scan_path: str, anno_path: str, category: int) -> None:
        super().__init__()
        self.scan_path = scan_path
        self.anno_path = anno_path
        self.category = category
        self.id = self.get_id()
        self.scan_object = nibabel.load(self.scan_path)
        self.anno_object = nibabel.load(self.anno_path)
        if (
            self.scan_object.header.get_data_shape()
            != self.anno_object.header.get_data_shape()
        ):
            raise ValueError(
                f"Data dimensions do not match: {self.scan_object.header.get_data_shape()} != {self.anno_object.header.get_data_shape()}"
            )

        if self.scan_object.header.get_zooms() != self.anno_object.header.get_zooms():
            raise ValueError(
                f"Voxel dimensions do not match: {self.scan_object.header.get_zooms()} != {self.anno_object.header.get_zooms()}"
            )

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

    def get_index_of_largest_slice(self):
        """
        Returns the largest slice of a scans' annotation.
        A "slice" has n-1 dimensions, where n is the number of dimensions of
        the original image.

        This will produce unexpected results for images with != 3 dimensions.
        """

        def get_slice_index(axis: int, index: int, dimensions: int = 3):
            slice_list: list = [slice(None) for _ in range(dimensions)]
            slice_list[axis] = slice(index, index + 1, None)
            return tuple(slice_list)

        anno_shape: list = self.anno_object.header.get_data_shape()
        anno_voxel_dimensions: list = self.anno_object.header.get_zooms()
        maximum_slice_area: float = 0
        maximum_slice_area_index: Optional[list] = None
        for dimension in range(len(self.anno_object.header.get_data_shape())):
            slice_voxel_dimensions: list = (
                anno_voxel_dimensions[:dimension]
                + anno_voxel_dimensions[dimension + 1 :]
            )
            slice_voxel_area: float = np.prod(slice_voxel_dimensions)
            for slice_number in range(anno_shape[dimension]):
                slice_index = get_slice_index(dimension, slice_number, len(anno_shape))
                voxel_slice: list = self.anno_object.get_fdata()[slice_index]
                nonzero_voxels: int = np.count_nonzero(voxel_slice)
                slice_area: float = nonzero_voxels * slice_voxel_area
                if slice_area > maximum_slice_area:
                    maximum_slice_area = slice_area
                    maximum_slice_area_index = slice_index
        return maximum_slice_area_index


class TestCase(TestCaseBase):
    """
    Represents a scan/annotation pair with health information. Offers cached
    functions.
    """

    def __init__(self, scan_path: str, anno_path: str, category: int) -> None:
        super().__init__(scan_path, anno_path, category)
        self.cached: dict = {}

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["cached"] = self.cached
        return dict_representation


if __name__ == "__main__":
    test_scan_path: str = "../Dataset_V2/MR81.nii.gz"
    test_save_path: str = "./81_save.txt"
    tc: TestCase = TestCase.from_scan_path(scan_path=test_scan_path, category=0)
    tc.to_file(test_save_path)
    tc2: TestCase = TestCase.from_file(test_save_path)

    print(tc.get_index_of_largest_slice())
    print(tc2.get_index_of_largest_slice())
