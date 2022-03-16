"""
Handles representation of MRI (scan/annotation) files in the form of NIfTI-1.
"""

from typing import Any, Callable, Optional
import json
import hashlib
import nibabel
import numpy as np
import os

import storable


class MRIFile(storable.Storable):
    """
    NIfTI-1 MRI file.
    """

    default_file_ending: str = ".nii.gz"

    path: str
    image: nibabel.Nifti1Image

    def __init__(self, path) -> None:
        self.path = path
        self.image = nibabel.load(self.path)

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["path"] = self.path
        dict_representation["id"] = self.get_id()
        return dict_representation

    def to_file(self, file_path: str):
        with open(file_path, "w") as output_file:
            json.dump(self.get_dict_representation(), output_file)

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as input_file:
            dict_representation: dict = json.load(input_file)
            return MRIFile.from_dict(dict_representation)

    @classmethod
    def from_dict(cls, dict_representation: dict):
        new_instance: MRIFile = cls(dict_representation["path"])
        if new_instance.get_id() != dict_representation["id"]:
            raise ValueError(
                "ID of loaded image does not match saved ID. The image file has been changed."
            )
        return new_instance

    def get_id(
        self, hash_function: Callable = hashlib.sha3_256, buffer_size: int = 4096
    ):
        """
        Consists of a file hash.
        """
        # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

        digestor: Any = hash_function()
        with open(self.path, "rb") as input_file:
            while True:
                data: bytes = input_file.read(buffer_size)
                if not data:
                    break
                digestor.update(data)
        return digestor.hexdigest()


class ScanFile(MRIFile):
    def __init__(self, path) -> None:
        super().__init__(path)


class AnnoFile(MRIFile):
    default_annotation_suffix: str = "A"

    def __init__(self, path) -> None:
        super().__init__(path)

    @classmethod
    def to_largest_slice(
        cls,
        path: str,
        anno_object: nibabel.Nifti1Image,
        overwrite: bool = False,
        use_existing: bool = True,
        along_axes: Optional[list] = None,
    ):
        """
        Creates new annotation from given by keeping only the largest slice.
        The new annotation is saved in path.
        """
        if not overwrite and os.path.exists(path):
            if use_existing:
                return AnnoFile(path)
            else:
                raise FileExistsError("File in path already exists.")
        largest_slice_anno: nibabel.Nifti1Image = AnnoFile._largest_slice_anno(
            anno_object, along_axes=along_axes
        )
        nibabel.save(largest_slice_anno, path)
        return AnnoFile(path)

    @staticmethod
    def get_slice_dimension(anno_object: nibabel.Nifti1Image):
        largest_slice_index = AnnoFile._get_largest_slice_index(anno_object)

        for slice_index, slice_object in enumerate(largest_slice_index):
            if slice_object != slice(None, None, None):
                return slice_index

    @staticmethod
    def _get_largest_slice_index(
        anno_object: nibabel.Nifti1Image, along_axes: Optional[list] = None
    ):
        """
        Returns the largest slice of a scans' annotation.
        A "slice" has n-1 dimensions, where n is the number of dimensions of
        the original image.

        This will produce unexpected results for images with != 3 dimensions.
        """

        def _get_slice_index(axis: int, index: int, dimensions: int = 3):
            slice_list: list = [slice(None) for _ in range(dimensions)]
            slice_list[axis] = slice(index, index + 1, None)
            return tuple(slice_list)

        anno_shape: list = anno_object.header.get_data_shape()
        anno_voxel_dimensions: list = anno_object.header.get_zooms()
        maximum_slice_area: float = 0
        maximum_slice_area_index: Optional[list] = None
        for dimension in range(len(anno_object.header.get_data_shape())):
            if along_axes is not None:
                if dimension not in along_axes:
                    continue
            slice_voxel_dimensions: list = (
                anno_voxel_dimensions[:dimension]
                + anno_voxel_dimensions[dimension + 1 :]
            )
            slice_voxel_area: float = np.prod(slice_voxel_dimensions)
            for slice_number in range(anno_shape[dimension]):
                slice_index = _get_slice_index(dimension, slice_number, len(anno_shape))
                voxel_slice: list = anno_object.get_fdata()[slice_index]
                nonzero_voxels: int = np.count_nonzero(voxel_slice)
                slice_area: float = nonzero_voxels * slice_voxel_area
                if slice_area > maximum_slice_area:
                    maximum_slice_area = slice_area
                    maximum_slice_area_index = slice_index
        return maximum_slice_area_index

    @staticmethod
    def _largest_slice_anno(
        anno_object: nibabel.Nifti1Image, along_axes: Optional[list] = None
    ):
        slice_anno_data = np.zeros(anno_object.header.get_data_shape())
        largest_slice_index: list = AnnoFile._get_largest_slice_index(
            anno_object, along_axes=along_axes
        )
        slice_anno_data[largest_slice_index] = anno_object.get_fdata()[
            largest_slice_index
        ]
        return nibabel.Nifti1Image(
            slice_anno_data, anno_object.affine, anno_object.header
        )

    @classmethod
    def from_scan_path(
        cls,
        scan_path: str,
        anno_suffix: Optional[str] = None,
        file_ending: Optional[str] = None,
    ):
        if anno_suffix is None:
            anno_suffix = AnnoFile.default_annotation_suffix
        if file_ending is None:
            file_ending = AnnoFile.default_file_ending
        no_ending: str
        if scan_path.endswith(file_ending):
            no_ending = scan_path[: -len(file_ending)]
        else:
            no_ending = scan_path
        anno_path: str = no_ending + anno_suffix + file_ending
        return cls(anno_path)
