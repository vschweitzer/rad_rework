from typing import List, Any, Dict, Union, Optional, Callable
import numpy as np
import sklearn
import sklearn.metrics as smetrics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import hashlib
import json

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
        classification_rounds: Optional[List[ClassificationRound]] = None,
    ) -> None:
        super().__init__()
        self.tcc = tcc
        self.fe = fe
        self.ff = ff

        self.train_set_size = train_set_size
        self.set_size_fractional = set_size_fractional
        self.metric = metric
        self.extractor_config_id = extractor_config_id
        if classification_rounds is None:
            self.classification_rounds: List[ClassificationRound] = []
        else:
            self.classification_rounds = classification_rounds

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

    def get_dict_representation(self, by_id: bool = True):
        dict_representation: dict = super().get_dict_representation(by_id=by_id)
        if by_id:
            dict_representation["tcc"] = self.tcc.get_id()
            dict_representation["fe"] = self.fe.get_id()
            dict_representation["ff"] = self.ff.get_id()
        else:
            dict_representation["tcc"] = self.tcc.get_dict_representation()
            dict_representation["fe"] = self.fe.get_dict_representation()
            dict_representation["ff"] = self.ff.get_dict_representation()
        dict_representation["classification_rounds"] = [
            r.get_dict_representation() for r in self.classification_rounds
        ]
        dict_representation["train_set_size"] = self.train_set_size
        dict_representation["metric"] = self.metric
        dict_representation["extractor_config_id"] = self.extractor_config_id

        return dict_representation

    def _save_dependencies(self, save_dir: str = "./"):
        super()._save_dependencies(save_dir)
        self.tcc.save(save_dir=save_dir, create=False)
        self.fe.save(save_dir=save_dir, create=False)
        self.ff.save(save_dir=save_dir, create=False)

    def get_feature_count(self):
        # Assumes equal feature count for every TestCase
        unfiltered_features: List[Dict[str, Any]] = self.fe.get_features(
            self.fe.get_configuration_id(),
            [list(self.tcc.test_cases.values())[0].get_id()],
        )
        filtered_features: List[Dict[str, Any]] = self.ff.filter(unfiltered_features)
        return len(list(filtered_features[0].keys()))

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

    def get_importance_average(
        self, weighted: bool = True, normalize: bool = False
    ) -> Dict[str, float]:
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

    def get_importance_distribution_plot(
        self, ax: plt.Axes, weighted: bool = True, normalize: bool = True
    ):
        importances: Dict[str, float] = self.get_importance_average(
            weighted=weighted, normalize=normalize
        )
        importance_values: List[float] = list(importances.values())
        sns.histplot(importance_values, bins=100, ax=ax)

    @staticmethod
    def get_accuracy_cascade_plot(
        ax: plt.Axes,
        classifications: List["Classification"],
        x_values: Optional[List[Any]] = None,
        x_title: Optional[str] = None,
        average_function: Callable[[Any], float] = np.mean,
        plot_feature_count: bool = True,
        logx: bool = True,
        cmap_name: str = "viridis"
    ):
        if x_values is None:
            x_values = list(range(len(classifications)))
        else:
            if len(x_values) != len(classifications):
                raise ValueError("Lengths of classifications and X values do not match")
        accuracies: List[float] = []
        feature_counts: List[int] = []
        separate_accuracies: List[List[float]] = []
        for c in classifications:
            round_accuracies: List[float] = c.get_balanced_accuracies()
            separate_accuracies.append(round_accuracies)
            round_accuracy: float = average_function(round_accuracies)
            accuracies.append(round_accuracy)
            feature_count: int = len(c.classification_rounds[0].feature_importances)
            feature_count_alternate: int = c.get_feature_count()
            assert feature_count == feature_count_alternate
            feature_counts.append(feature_count)
        separate_accuracy_values = list(zip(*separate_accuracies))
        colormap = matplotlib.cm.get_cmap(cmap_name)
        for index, accuracy_values in enumerate(separate_accuracy_values):
            sns.lineplot(ax=ax, x=feature_counts, y=accuracy_values, color=colormap(index/len(separate_accuracy_values)))
        
        sns.scatterplot(x=feature_counts, y=accuracies, ax=ax, color="#8800ff")

        ax.set_ylabel(f"Accuracy {average_function.__name__}")
        ax.set_xlabel(f"Feature Count")
        if logx:
            ax.set_xscale("log")
        steps: int = len(classifications)
        rounds_each: float = np.average(
            [len(rounds.classification_rounds) for rounds in classifications]
        )
        accuracy_title: str = f"Accuracy over {steps} steps, {rounds_each} rounds each."
        ax.set_title(accuracy_title)

    @staticmethod
    def get_cascade_accuracies(
        classifications: list,
        adjusted: bool = False,
        average_function: Callable[[list], float] = np.mean,
    ):
        average_accuracies: List[float] = []

        for c in classifications:
            accuracies = c.get_balanced_accuracies(adjusted=adjusted)
            average_accuracy: float = average_function(accuracies)
            average_accuracies.append(average_accuracy)
        return average_accuracies

    @staticmethod
    def save_collection(
        classifications: List["Classification"],
        save_dir: str = "./",
        hash_function: Callable = hashlib.sha3_256,
        buffer_size: int = 4096,
    ):
        for c in classifications:
            c.save(save_dir=save_dir)

        id_list: List[str] = [c.get_id() for c in classifications]

        # Hash combination of IDs to get list ID
        to_hash: str = "".join(id_list)

        digestor: Any = hash_function()
        digestor.update(to_hash.encode("utf-8"))
        list_id: str = digestor.hexdigest()
        save_path: str = os.path.join(save_dir, list_id + ".json")
        with open(save_path, "w") as save_file:
            json.dump(id_list, save_file)
