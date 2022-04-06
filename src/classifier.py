import copy
from typing import Callable, List, Any, Dict, Union
import numpy as np
import sklearn.ensemble as skensemble

import storable
import test_case_collection
import feature_filter
import feature_extractor
import classification


class Classifier(storable.Storable):
    classifier: skensemble.RandomForestClassifier
    random_seed: int

    def __init__(
        self, random_seed: int = 0, classifier_options: Dict[str, Any] = {}
    ) -> None:
        super().__init__()
        self.random_seed = random_seed
        self.classifier_options = classifier_options
        self.classifier = skensemble.RandomForestClassifier(
            **self.classifier_options, random_state=self.random_seed
        )

    def fit(self, features: List[List[Any]], target_values: List[int]):
        self.classifier.fit(features, target_values)

    def predict(self, features: List[List[Any]]):
        return self.classifier.predict(features)

    def classify(
        self,
        tcc: test_case_collection.TestCaseCollection,
        fe: feature_extractor.FeatureExtractor,
        ff: feature_filter.FeatureFilter,
        rounds: int = 100,
        train_set_size: Union[int, float] = 0.7,
        set_size_fractional: bool = True,
        metric: str = "nar",
    ):
        extractor_config_id: str = fe.get_configuration_id()
        classification_result: classification.Classification = (
            classification.Classification(
                tcc,
                fe,
                ff,
                train_set_size,
                set_size_fractional,
                metric,
                extractor_config_id,
            )
        )
        for round in range(rounds):
            print(f"Round: {round}")
            round_seed: int = self.random_seed + round
            cf: Classifier = Classifier(
                round_seed, classifier_options=self.classifier_options
            )

            train_set: List[str] = tcc.get_equal_sample(
                count=train_set_size, fractional=set_size_fractional, metric=metric
            )
            test_set: List[str] = tcc.get_all_but(train_set)
            test_set = tcc.remove_none_values(test_set, metric=metric)

            train_targets = tcc.get_target_values(train_set, metric=metric)
            test_targets = tcc.get_target_values(test_set, metric=metric)

            train_features: List[list] = fe.to_sklearn_data(
                ff.filter(fe.get_features(extractor_config_id, train_set))
            )
            test_features: List[list] = fe.to_sklearn_data(
                ff.filter(fe.get_features(extractor_config_id, test_set))
            )

            cf.fit(train_features, train_targets)
            predictions = cf.predict(test_features)

            classification_result.classification_rounds.append(
                classification.ClassificationRound(
                    train_set,
                    test_set,
                    round_seed,
                    predictions,
                    cf.classifier.feature_importances_,
                )
            )

        return classification_result

    def importance_cascade(
        self,
        tcc: test_case_collection.TestCaseCollection,
        fe: feature_extractor.FeatureExtractor,
        ff: feature_filter.FeatureFilter,
        rounds: int = 100,
        train_set_size: Union[int, float] = 0.7,
        set_size_fractional: bool = True,
        metric: str = "nar",
        steps: int = 100,
    ):
        base_classification = self.classify(
            tcc,
            fe,
            ff,
            rounds=rounds,
            train_set_size=train_set_size,
            set_size_fractional=set_size_fractional,
            metric=metric,
        )
        base_importances: Dict[str, Any] = base_classification.get_importance_average(
            weighted=True, normalize=True
        )
        max_importance: float = max(base_importances.values())
        classifications: List[classification.Classification] = []
        for step in range(steps):
            threshold: float = step / steps * max_importance
            print(f"Round {step}, Threshold: {threshold}")
            step_filter: feature_filter.FeatureFilter = feature_filter.FeatureFilter(
                "importance_threshold",
                base_importances,
                threshold,
                subfilters=[ff],
            )

            result = self.classify(
                tcc,
                fe,
                step_filter,
                rounds=rounds,
                train_set_size=train_set_size,
                set_size_fractional=set_size_fractional,
                metric=metric,
            )
            classifications.append(result)
        return classifications
