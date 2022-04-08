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


class TestCase(storable.Storable):
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

    def get_dict_representation(self, by_id: bool = True):
        dict_representation: dict = super().get_dict_representation(by_id=by_id)
        if by_id:
            dict_representation["scan_file"] = self.scan_file.get_id()
            dict_representation["anno_file"] = self.anno_file.get_id()
        else:
            dict_representation["scan_file"] = self.scan_file.get_dict_representation()
            dict_representation["anno_file"] = self.anno_file.get_dict_representation()

        dict_representation["pcr"] = self.pcr
        dict_representation["nar"] = self.nar
        dict_representation["id"] = self.id
        return dict_representation

    def _save_dependencies(self, save_dir: str = "./"):
        super()._save_dependencies(save_dir)
        self.scan_file.save(save_dir=save_dir, create=False)
        self.anno_file.save(save_dir=save_dir, create=False)

    def to_file(self, file_path: str):
        with open(file_path, "w") as output_file:
            json.dump(self.get_dict_representation(by_id=False), output_file)

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

    def to_2D(
        self,
        new_annotation_path: Optional[str] = None,
        overwrite: bool = False,
        use_existing: bool = True,
        along_axes: Optional[list] = None,
    ):
        new_annotation: mri_file.AnnoFile
        if new_annotation_path is None:
            no_ending: str
            if self.anno_file.path.endswith(self.anno_file.default_file_ending):
                no_ending = self.anno_file.path[
                    : -len(self.anno_file.default_file_ending)
                ]
            else:
                no_ending = self.anno_file.path
            new_annotation_path = no_ending + "_largest_slice_"
            if along_axes is None:
                new_annotation_path += "axis_any" + self.anno_file.default_file_ending
            else:
                new_annotation_path += (
                    "_".join([str(element) for element in along_axes])
                    + self.anno_file.default_file_ending
                )

        new_annotation = mri_file.AnnoFile.to_largest_slice(
            new_annotation_path,
            self.anno_file.image,
            overwrite=overwrite,
            use_existing=use_existing,
            along_axes=along_axes,
        )
        self.anno_file = new_annotation
        self.id = self.get_id()


if __name__ == "__main__":
    test_scan_path: str = "../Dataset_V2/MR86.nii.gz"
    test_save_path: str = "./86_save.txt"
    tc: TestCase = TestCase.from_scan_path(scan_path=test_scan_path, pcr=True, nar=1)
    tc.to_file(test_save_path)
    tc2: TestCase = TestCase.from_file(test_save_path)
    tc2.to_2D(overwrite=True, use_existing=False, along_axes=[0])
    print(tc)
    print(tc2)
