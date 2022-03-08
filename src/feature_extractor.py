import radiomics.featureextractor
import json

import test_case
import storable


class FeatureExtractor(storable.Storable):
    """
    Wrapper for radiomics.featureextractor.RadiomicsFeatureExtractor.
    Stores config and per-config/per-TestCase results.
    """

    config: dict
    features: dict
    extractor: radiomics.featureextractor.RadiomicsFeatureExtractor

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.features = {}
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.extractor.loadJSONParams(json.dumps(self.config))

    def extract(self, tc: test_case.TestCase):
        """
        Extracts radiomics features from TestCase. These features are stored in
        self.features[<config_id>][<TestCase_id>].
        """
        pass

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


if __name__ == "__main__":
    fe = FeatureExtractor({})
    print(fe)
