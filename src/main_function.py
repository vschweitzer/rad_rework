import json
import matplotlib.pyplot as plt
from typing import Any, Dict, List
from sklearn.metrics import ConfusionMatrixDisplay
import os
import seaborn as sns
import datetime

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

    tcc.convert_annotations(conversion_options={"along_axes": [2]})
    fe.extract_collection(tcc)
    fe.to_file(extractor_save_path)

    steps: int = 10
    rounds: int = 5
    classifications_importance: List[classification.Classification] = cf.random_cascade(
        tcc, fe, ff, steps=steps, rounds=rounds, metric=metric
    )
    classifications_random: List[classification.Classification] = cf.random_cascade(
        tcc, fe, ff, steps=steps, rounds=rounds, metric=metric
    )
    # classifications_serializable = [
    #     c.get_dict_representation() for c in classifications
    # ]
    # timestamp: str = datetime.datetime.strftime(
    #     datetime.datetime.now(), "%Y%m%d_%H%M%S"
    # )
    # with open(f"classifications_{timestamp}.json", "w") as classifications_save:
    #     json.dump(classifications_serializable, classifications_save)

    fig = plt.figure()
    subfigs = fig.subfigures(nrows=2, ncols=1)
    fig.suptitle(f"{metric.upper()} classification")
    subtitles: List[str] = ["Importance Cascade", "Random Cascade"]
    axs: List = []
    for index, (subfigure, subtitle) in enumerate(zip(subfigs, subtitles)):
        subfigure.suptitle(subtitle)
        axs.append(subfigure.subplots(nrows=1, ncols=2))
    classification.Classification.get_accuracy_cascade_plot(
        axs[0][0], classifications_importance
    )
    classifications_importance[0].get_importance_distribution_plot(axs[0][1])

    classification.Classification.get_accuracy_cascade_plot(
        axs[1][0], classifications_random
    )
    classifications_random[0].get_importance_distribution_plot(axs[1][1])
    plt.show()

    pass


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
