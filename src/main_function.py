import json
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from sklearn.metrics import ConfusionMatrixDisplay
import os
import seaborn as sns

import test_case
import test_case_collection
import feature_extractor
import feature_filter
import classifier
import classification


def load_extract_and_filter(
    tcc_path: str = "../Dataset_V2/images_clean_NAR.csv", extractor_config: dict = {}
):
    tcc: test_case_collection.TestCaseCollection = (
        test_case_collection.TestCaseCollection.from_csv(tcc_path)
    )
    fe: feature_extractor.FeatureExtractor = feature_extractor.FeatureExtractor(
        extractor_config, force2D_adaptive_axis=False
    )

    extractor_config_id: str = fe.get_configuration_id()
    extractor_save_path: str = extractor_config_id + ".json"
    if os.path.exists(extractor_save_path):
        fe = feature_extractor.FeatureExtractor.from_file(extractor_save_path)

    ff: feature_filter.FeatureFilter = feature_filter.FeatureFilter(
        "key_starts_with", "diagnostics_", invert=True
    )
    cf: classifier.Classifier = classifier.Classifier(
        random_seed=0, classifier_options={"n_estimators": 1000, "n_jobs": 8}
    )
    metric: str = "pcr"
    metric_possibilities: int
    if metric == "nar":
        metric_possibilities = 3
    elif metric == "pcr":
        metric_possibilities = 2
    else:
        raise ValueError("Unknown metric")

    tcc.convert_annotations(conversion_options={"along_axes": [2]})
    fe.extract_collection(tcc)
    fe.to_file(extractor_save_path)

    steps: int = 100
    classifications: List[classification.Classification] = cf.importance_cascade(
        tcc, fe, ff, steps=steps, rounds=10
    )
    classifications_serializable = [
        c.get_dict_representation() for c in classifications
    ]
    with open("classifications.json", "w") as classifications_save:
        json.dump(classifications_serializable, classifications_save)
    accuracies = classification.Classification.get_cascade_accuracies(classifications)
    sns.lineplot(x=range(steps), y=accuracies)
    plt.show()


if __name__ == "__main__":
    load_extract_and_filter(
        extractor_config={
            "imageType": {
                "Original": {},
                "LoG": {},
                "Wavelet": {},
                "Square": {},
                "SquareRoot": {},
                "Logarithm": {},
                "Exponential": {},
                "Gradient": {},
                "LBP2D": {},
            },
            "featureClass": {
                "shape2D": None,
                "firstorder": None,
                "glcm": [
                    "Autocorrelation",
                    "JointAverage",
                    "ClusterProminence",
                    "ClusterShade",
                    "ClusterTendency",
                    "Contrast",
                    "Correlation",
                    "DifferenceAverage",
                    "DifferenceEntropy",
                    "DifferenceVariance",
                    "JointEnergy",
                    "JointEntropy",
                    "Imc1",
                    "Imc2",
                    "Idm",
                    "Idmn",
                    "Id",
                    "Idn",
                    "InverseVariance",
                    "MaximumProbability",
                    "SumEntropy",
                    "SumSquares",
                ],
                "glrlm": None,
                "glszm": None,
                "gldm": None,
                "ngtdm": None,
            },
            "setting": {"correctMask": True, "force2D": True, "force2Ddimension": 0},
        }
    )
