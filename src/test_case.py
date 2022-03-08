""" 
Handles the representation of patients/test cases
"""

from typing import Any, Callable, Optional
import json
import hashlib
import nibabel
import numpy as np

import storable
import mri_file


class TestCaseBase(storable.Storable):
    """
    Represents a scan/annotation pair with health information.
    """

    # stored
    scan_file: mri_file.ScanFile
    anno_file: mri_file.AnnoFile

    nar: Optional[int]
    pcr: Optional[bool]
    id: str

    # not stored
    anno_object: nibabel.Nifti1Image
    scan_object: nibabel.Nifti1Image

    def __init__(
        self,
        scan_file: mri_file.ScanFile,
        anno_file: mri_file.AnnoFile,
        pcr: Optional[bool],
        nar: Optional[int],
    ) -> None:
        super().__init__()
        self.scan_file = scan_file
        self.anno_file = anno_file
        self.pcr = pcr
        self.nar = nar
        self.id = self.get_id()

        if (
            self.scan_file.image.header.get_data_shape()
            != self.anno_file.image.header.get_data_shape()
        ):
            raise ValueError(
                f"Data dimensions do not match: {self.scan_file.image.header.get_data_shape()} != {self.anno_file.image.header.get_data_shape()}"
            )

        if (
            self.scan_file.image.header.get_zooms()
            != self.anno_file.image.header.get_zooms()
        ):
            raise ValueError(
                f"Voxel dimensions do not match: {self.scan_file.image.header.get_zooms()} != {self.anno_file.image.header.get_zooms()}"
            )

    def get_id(
        self, hash_function: Callable = hashlib.sha3_256, buffer_size: int = 4096
    ):
        """
        Consists of <scan_hash>_<anno_hash>.
        """
        # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

        return (
            self.scan_file.get_id(hash_function=hash_function, buffer_size=buffer_size)
            + "_"
            + self.anno_file.get_id(
                hash_function=hash_function, buffer_size=buffer_size
            )
        )

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["scan_file"] = self.scan_file.get_dict_representation()
        dict_representation["anno_file"] = self.anno_file.get_dict_representation()
        dict_representation["pcr"] = self.pcr
        dict_representation["nar"] = self.nar
        dict_representation["id"] = self.id
        return dict_representation

    def to_file(self, file_path: str):
        with open(file_path, "w") as output_file:
            json.dump(self.get_dict_representation(), output_file)

    @classmethod
    def from_file(cls, file_path: str):
        dict_representation: dict = {}
        with open(file_path, "r") as input_file:
            dict_representation = json.load(input_file)

        scan_file: mri_file.ScanFile = mri_file.ScanFile.from_dict(
            dict_representation["scan_file"]
        )
        anno_file: mri_file.AnnoFile = mri_file.AnnoFile.from_dict(
            dict_representation["anno_file"]
        )

        return cls(
            scan_file, anno_file, dict_representation["pcr"], dict_representation["nar"]
        )

    @classmethod
    def from_scan_path(
        cls,
        scan_path: str,
        pcr: Optional[bool],
        nar: Optional[int],
        anno_suffix: Optional[str] = None,
        file_ending: Optional[str] = None,
    ):
        """
        Creates TestCase. Annotation path is inferred from the given scan path.

        None for either anno_suffix or file_ending will lead to their default
        value being used.
        """
        scan_file: mri_file.ScanFile = mri_file.ScanFile(scan_path)
        anno_file: mri_file.AnnoFile = mri_file.AnnoFile.from_scan_path(
            scan_path, anno_suffix=anno_suffix, file_ending=file_ending
        )
        return cls(scan_file, anno_file, pcr, nar)


class TestCase(TestCaseBase):
    """
    Represents a scan/annotation pair with health information. Offers cached
    functions.
    """

    def __init__(
        self,
        scan_file: mri_file.ScanFile,
        anno_file: mri_file.AnnoFile,
        pcr: Optional[bool],
        nar: Optional[int],
    ) -> None:
        super().__init__(scan_file, anno_file, pcr, nar)
        self.cached: dict = {}

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["cached"] = self.cached
        return dict_representation


if __name__ == "__main__":
    test_scan_path: str = "../Dataset_V2/MR81.nii.gz"
    test_save_path: str = "./81_save.txt"
    tc: TestCase = TestCase.from_scan_path(scan_path=test_scan_path, pcr=True, nar=1)
    tc.to_file(test_save_path)
    tc2: TestCase = TestCase.from_file(test_save_path)

    print(tc)
    print(tc2)
