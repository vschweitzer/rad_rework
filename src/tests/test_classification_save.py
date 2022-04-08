import os
from typing import List

import import_parent
import classifier
import classification
import test_case_collection
import feature_extractor
import feature_filter


def test_classification_save(
    tcc_path: str = "../Dataset_V2/images_clean_NAR.csv", extractor_config: dict = {}
):
    tcc: test_case_collection.TestCaseCollection = (
        test_case_collection.TestCaseCollection.from_csv(tcc_path)
    )
    fe: feature_extractor.FeatureExtractor = feature_extractor.FeatureExtractor(
        extractor_config, force2D_adaptive_axis=False
    )

    ff: feature_filter.FeatureFilter = feature_filter.FeatureFilter(
        "key_starts_with", "diagnostics_", invert=True
    )
    cf: classifier.Classifier = classifier.Classifier(
        random_seed=0, classifier_options={"n_estimators": 1000, "n_jobs": 8}
    )

    tcc.convert_annotations(conversion_options={"along_axes": [2]})
    fe.extract_collection(tcc)

    classification_result: classification.Classification = cf.classify(
        tcc, fe, ff, rounds=1, metric="nar"
    )
    classification_result.save(save_dir="./saves", create=True)


if __name__ == "__main__":
    test_classification_save(
        extractor_config={
            "imageType": {
                "Original": {},
            },
            "setting": {"correctMask": True, "force2D": True, "force2Ddimension": 0},
        }
    )
