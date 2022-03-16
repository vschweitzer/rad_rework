import radiomics.featureextractor
import json
import numpy as np
from typing import Any, Callable
import os

from mri_file import AnnoFile
import hashlib
import storable
import test_case
import test_case_collection


class FeatureExtractor(storable.Storable):
    """
    Wrapper for radiomics.featureextractor.RadiomicsFeatureExtractor.
    Stores config and per-config/per-TestCase results.
    """

    config: dict
    features: dict
    extractor: radiomics.featureextractor.RadiomicsFeatureExtractor
    force2D_adaptive_axis: bool

    def __init__(self, config: dict, force2D_adaptive_axis: bool = True) -> None:
        """
        Accepts a dictionary containing a configuration for the
        radiomics.featureextractor.RadiomicsFeatureExtractor.

        If force2D_adaptive_axis is True, the extractor will check if force2D
        is enabled. If so, the value for force2Ddimension will be determined
        automatically per TestCase.
        """
        super().__init__()
        self.config = config
        self.features = {}
        self.force2D_adaptive_axis = force2D_adaptive_axis
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.extractor.loadJSONParams(json.dumps(self.config))

    def get_configuration_id(self, hash_function: Callable = hashlib.sha3_256):
        config_str: str = json.dumps(self.config, sort_keys=True)
        config_bytes: bytes = config_str.encode("utf-8")
        digestor: Any = hash_function()

        data: bytes = config_bytes
        digestor.update(data)
        return digestor.hexdigest()

    def extract_collection(self, tcc: test_case_collection.TestCaseCollection):
        for tc in tcc.test_cases:
            new_features = self.extract(tc)
            json.dumps(new_features)

    def extract(self, tc: test_case.TestCase):
        """
        Extracts radiomics features from TestCase. These features are stored in
        self.features[<config_id>][<TestCase_id>].
        """
        features: dict
        converted_features: dict

        dimension_2D: int = AnnoFile.get_slice_dimension(tc.anno_file.image)
        if self.force2D_adaptive_axis:
            if (
                "force2D" in self.config["setting"]
                and self.config["setting"]["force2D"]
            ):
                self.change_setting("force2Ddimension", dimension_2D)
            else:
                self.remove_setting("force2Ddimension")

        config_id: str = self.get_configuration_id()

        if config_id in self.features and tc.id in self.features[config_id]:
            converted_features = self.features[config_id][tc.id]
        else:
            features = self.extractor.execute(tc.scan_file.path, tc.anno_file.path)
            converted_features = FeatureExtractor.denumpyify(features)

            if config_id not in self.features:
                self.features[config_id] = {}
            self.features[config_id][tc.id] = converted_features
        return converted_features

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["config"] = self.config
        dict_representation["features"] = self.features
        return dict_representation

    def to_file(self, file_path: str):
        with open(file_path, "w") as output_file:
            json.dump(self.get_dict_representation(), output_file)

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as input_file:
            dict_representation: dict = json.load(input_file)
        return FeatureExtractor.from_dict(dict_representation)

    @classmethod
    def from_dict(cls, dict_representation: dict):
        fe: FeatureExtractor = cls(dict_representation["config"])
        fe.features = dict_representation["features"]
        return fe

    @staticmethod
    def denumpyify(variable: Any):
        if isinstance(variable, np.ndarray):
            return variable.tolist()
        elif isinstance(variable, np.float64):
            return float(variable)
        elif isinstance(variable, dict):
            new_dict: dict = {}
            for key in variable:
                new_key: Any = FeatureExtractor.denumpyify(key)
                new_value: Any = FeatureExtractor.denumpyify(variable[key])
                new_dict[new_key] = new_value
            return new_dict
        elif isinstance(variable, (list, tuple)):
            new_list: list = []
            for element in variable:
                new_element: Any = FeatureExtractor.denumpyify(element)
                new_list.append(new_element)
            return new_list
        else:
            return variable

    def change_setting(self, key: str, value: Any):
        temporary_settings: dict = self.config
        temporary_settings["setting"] = self.extractor.settings
        temporary_settings["setting"][key] = value
        self.extractor.loadJSONParams(json.dumps(temporary_settings))
        self.config = temporary_settings

    def remove_setting(self, key):
        temporary_settings: dict = self.config
        temporary_settings["setting"] = self.extractor.settings
        temporary_settings["setting"].pop(key)
        self.extractor.loadJSONParams(json.dumps(temporary_settings))
        self.config = temporary_settings


if __name__ == "__main__":
    import pprint

    save_path: str = "extractor.json"
    if os.path.exists(save_path):
        fe = FeatureExtractor.from_file(save_path)
    else:
        fe = FeatureExtractor(
            {"setting": {"correctMask": True, "force2D": True, "force2Ddimension": 0}},
            force2D_adaptive_axis=False,
        )

    pp = pprint.PrettyPrinter(indent=4)
    tcc: test_case_collection.TestCaseCollection = (
        test_case_collection.TestCaseCollection.from_csv(
            "../Dataset_V2/images_clean_NAR.csv"
        )
    )
    tcc.convert_annotations(conversion_options={"along_axes": [2]})
    fe.extract_collection(tcc)
    pp.pprint(fe.get_dict_representation())
    fe.to_file(save_path)
