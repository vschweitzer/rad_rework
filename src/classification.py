from typing import List, Any, Dict, Union, Optional
import numpy as np
import sklearn
import sklearn.metrics as smetrics

import storable
import test_case_collection
import feature_filter
import feature_extractor
import test_case


class ClassificationRound(storable.Storable):
    def __init__(
        self,
        train_set: List[str],
        test_set: List[str],
        random_seed: int,
        predictions: List[Any],
        feature_importances: List[float],
    ) -> None:
        super().__init__()
        self.train_set = train_set
        self.test_set = test_set
        self.random_seed = random_seed
        self.predictions = feature_extractor.FeatureExtractor.denumpyify(predictions)
        self.feature_importances = feature_extractor.FeatureExtractor.denumpyify(
            feature_importances
        )

    def get_dict_representation(self):
        dict_representation = super().get_dict_representation()
        dict_representation["train_set"] = self.train_set
        dict_representation["test_set"] = self.test_set
        dict_representation["random_seed"] = self.random_seed
        dict_representation["predictions"] = self.predictions
        dict_representation["feature_importances"] = self.feature_importances
        return dict_representation

    @classmethod
    def from_dict(cls, dict_representation: dict):
        cls(
            dict_representation["train_set"],
            dict_representation["test_set"],
            dict_representation["random_seed"],
            dict_representation["predictions"],
            dict_representation["feature_importances"],
        )


class Classification(storable.Storable):
    predictions: List[Any]

    tcc: test_case_collection.TestCaseCollection
    fe: feature_extractor.FeatureExtractor
    ff: feature_filter.FeatureFilter

    random_seed: int
    train_set_size: Union[int, float]
    set_size_fractional: bool
    metric: str
    extractor_config_id: str

    def __init__(
        self,
        tcc: test_case_collection.TestCaseCollection,
        fe: feature_extractor.FeatureExtractor,
        ff: feature_filter.FeatureFilter,
        train_set_size: Union[int, float],
        set_size_fractional: bool,
        metric: str,
        extractor_config_id: str,
        classification_rounds: List[ClassificationRound] = [],
    ) -> None:
        super().__init__()
        self.tcc = tcc
        self.fe = fe
        self.ff = ff

        self.train_set_size = train_set_size
        self.set_size_fractional = set_size_fractional
        self.metric = metric
        self.extractor_config_id = extractor_config_id

        self.classification_rounds: List[ClassificationRound] = classification_rounds

    @classmethod
    def from_dict(cls, dict_representation: dict):
        return cls(
            test_case_collection.TestCaseCollection.from_dict(
                dict_representation["tcc"]
            ),
            feature_extractor.FeatureExtractor.from_dict(dict_representation["fe"]),
            feature_filter.FeatureFilter.from_dict(dict_representation["ff"]),
            dict_representation["train_set_size"],
            dict_representation["set_size_fractional"],
            dict_representation["metric"],
            dict_representation["extractor_config_id"],
            [
                ClassificationRound.from_dict(r)
                for r in dict_representation["classification_rounds"]
            ],
        )

    def get_dict_representation(self):
        dict_representation = super().get_dict_representation()
        dict_representation["tcc"] = self.tcc.get_dict_representation()
        dict_representation["fe"] = self.fe.get_dict_representation()
        dict_representation["ff"] = self.ff.get_dict_representation()
        dict_representation["train_set_size"] = self.train_set_size
        dict_representation["metric"] = self.metric
        dict_representation["extractor_config_id"] = self.extractor_config_id
        dict_representation["classification_rounds"] = [
            r.get_dict_representation() for r in self.classification_rounds
        ]
        return dict_representation

    def get_confusion_matrix(self):
        total_targets: List[Any] = []
        total_predictions: List[Any] = []
        for classification_round in self.classification_rounds:
            targets = self.tcc.get_target_values(
                classification_round.test_set, self.metric
            )
            predictions = classification_round.predictions
            total_targets.append(targets)
            total_predictions.append(predictions)

        smetrics.confusion_matrix(total_targets, total_predictions)

    def get_balanced_accuracies(self, adjusted: bool = False) -> List[float]:
        accuracies: List[float] = []
        for classification_round in self.classification_rounds:
            targets = self.tcc.get_target_values(
                classification_round.test_set, self.metric
            )
            predictions = classification_round.predictions
            accuracy: float = smetrics.balanced_accuracy_score(
                targets, predictions, adjusted=adjusted
            )
            accuracies.append(accuracy)
        return accuracies

    def get_importance_average(self, weighted: bool = True, normalize: bool = False):
        total_importances: List[List[float]] = []
        first_test_case: str = list(self.fe.features[self.extractor_config_id].keys())[
            0
        ]
        unfiltered_features: List[Dict[str, Any]] = [
            self.fe.features[self.extractor_config_id][first_test_case]
        ]
        filtered_features: List[Dict[str, Any]] = self.ff.filter(unfiltered_features)
        feature_names: List[str] = list(filtered_features[0].keys())
        weights: Optional[List[float]] = None
        for classification_round in self.classification_rounds:
            round_importances: List[float] = classification_round.feature_importances
            total_importances.append(round_importances)

        if weighted:
            weights = self.get_balanced_accuracies(adjusted=True)

        average_importances = np.average(
            np.array(total_importances), axis=0, weights=weights
        )
        if normalize:
            correction_factor: float = np.sum(average_importances)
            average_importances /= correction_factor

        return dict(zip(feature_names, average_importances))
