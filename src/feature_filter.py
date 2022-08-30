import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import storable
import copy


class FeatureFilter(storable.Storable):
    """
    FeatureFilters accept a list of feature dictionaries.
    """

    args: tuple
    kwargs: dict
    filter_name: str
    subfilters: list
    _filter_prefix: str = "_filter_"

    def __init__(
        self, filter_name: str, *args, subfilters: Optional[list] = None, **kwargs
    ) -> None:
        super().__init__()
        if filter_name not in self.get_available_filters():
            raise ValueError("Unrecognized filter")

        if subfilters is None:
            self.subfilters = []
        else:
            self.subfilters = copy.deepcopy(subfilters)
        self.filter_name = filter_name
        self.args = copy.deepcopy(args)
        self.kwargs = copy.deepcopy(kwargs)

    @classmethod
    def from_dict(cls, dict_representation: dict):
        return cls(
            dict_representation["filter_name"],
            *dict_representation["args"],
            subfilters=dict_representation["subfilters"],
            **dict_representation["kwargs"]
        )

    def get_dict_representation(self, by_id: bool = True):
        dict_representation: dict = super().get_dict_representation(by_id=by_id)
        dict_representation["is_filter"] = True
        dict_representation["filter_name"] = self.filter_name

        if by_id:
            dict_representation["subfilters"] = [
                subfilter.get_id() for subfilter in self.subfilters
            ]
        else:
            dict_representation["subfilters"] = [
                subfilter.get_dict_representation() for subfilter in self.subfilters
            ]
        dict_representation["args"] = self.args
        dict_representation["kwargs"] = self.kwargs
        return dict_representation

    def _save_dependencies(self, save_dir: str = "./"):
        super()._save_dependencies()
        for subfilter in self.subfilters:
            subfilter.save(save_dir=save_dir, create=False)

    def execute_subfilters(
        self, features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.subfilters:
            return features

        filtered_features: List[Dict[str, Any]] = []

        for subfilter in self.subfilters:
            subfilter_result = subfilter.filter(features)
            if not filtered_features:
                filtered_features = subfilter_result
            else:
                for old_features, new_features in zip(
                    filtered_features, subfilter_result
                ):
                    old_features.update(new_features)
        return filtered_features

    def filter(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        subfiltered_features: List[Dict[str, Any]] = self.execute_subfilters(features)
        result = getattr(self, self._filter_prefix + self.filter_name)(
            subfiltered_features, *self.args, **self.kwargs
        )

        if "random_seed" in self.kwargs and "increase_seed" in self.kwargs:
            if self.kwargs["increase_seed"]:
                self.kwargs["random_seed"] += 1

        return result

    def get_available_filters(self):
        available_filters: list = [
            func[len(self._filter_prefix) :]
            for func in dir(self)
            if callable(getattr(self, func)) and func.startswith(self._filter_prefix)
        ]
        return available_filters

    @staticmethod
    def _filter_do_nothing(
        features: List[Dict[str, Any]], *args, **kwargs
    ) -> List[Dict[str, Any]]:
        return features

    @staticmethod
    def _filter_key_starts_with(
        features: List[Dict[str, Any]],
        prefix: str,
        *args,
        invert: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        filtered_features: List[Dict[str, Any]] = []
        for feature_dict in features:
            filtered_dict: Dict[str, Dict[str, Any]] = {}
            for key in feature_dict:
                if key.startswith(prefix) != invert:
                    filtered_dict[key] = feature_dict[key]
            filtered_features.append(filtered_dict)
        return filtered_features

    @staticmethod
    def _filter_importance_threshold(
        features: List[Dict[str, Any]],
        importances: Dict[str, Any],
        threshold: float,
        *args,
        invert: bool = False,
        **kwargs
    ):
        """
        Returns all features with larger or equal importance
        """
        filtered_features: List[Dict[str, Any]] = copy.deepcopy(features)
        for feature in importances:
            remove_feature: bool = False
            if importances[feature] < threshold:
                remove_feature = True

            if invert:
                remove_feature = not remove_feature

            if remove_feature:
                for test_case in filtered_features:
                    test_case.pop(feature)

        return filtered_features

    @staticmethod
    def _filter_random_choice(
        features: List[Dict[str, Any]],
        fraction: float,
        *args,
        random_seed: int = 0,
        invert: bool = False,
        increase_seed: bool = True,
        **kwargs
    ):
        if not features:
            return features

        filtered_features: List[Dict[str, Any]] = []
        to_choose: int = round(len(features[0].keys()) * fraction)
        random_provider: random.Random = random.Random(random_seed)
        all_keys: Set[str] = set(features[0].keys())
        key_selection: Set[str] = set(random_provider.sample(all_keys, to_choose))
        if invert:
            all_keys.difference_update(key_selection)
            key_selection = all_keys

        for feature_set in features:
            filtered_feature_set = {key: feature_set[key] for key in key_selection}
            filtered_features.append(filtered_feature_set)

        return filtered_features


if __name__ == "__main__":
    import pprint

    pp: pprint.PrettyPrinter = pprint.PrettyPrinter()
    test_features = [
        {
            "key_1": "value_1",
            "key_2": "value_2",
            "key_3": "value_3",
            "key_2": "value_2",
            "diagnostics_1": "value_1",
            "diagnostics_2": "value_2",
            "diagnostics_3": "value_3",
        },
        {
            "key_1": "another_value_1",
            "key_2": "another_value_2",
            "key_3": "another_value_3",
            "key_2": "another_value_2",
            "diagnostics_1": "value_1",
            "diagnostics_2": "value_2",
            "diagnostics_3": "value_3",
        },
        {
            "key_1": "still_another_value_1",
            "key_2": "still_another_value_2",
            "key_3": "still_another_value_3",
            "key_2": "still_another_value_2",
            "diagnostics_1": "diagnostic_value_1",
            "diagnostics_2": "diagnostic_value_2",
            "diagnostics_3": "diagnostic_value_3",
        },
    ]

    subfilter: FeatureFilter = FeatureFilter("key_starts_with", "key")
    ff: FeatureFilter = FeatureFilter(
        "key_starts_with", "key_1", subfilters=[subfilter], invert=True
    )
    pp.pprint(ff.filter(test_features))
