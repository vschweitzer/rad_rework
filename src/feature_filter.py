from typing import Any, Callable, Dict, List, Tuple

import storable


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
        self, filter_name: str, *args, subfilters: list = [], **kwargs
    ) -> None:
        super().__init__()
        if filter_name not in self.get_available_filters():
            raise ValueError("Unrecognized filter")

        self.subfilters = subfilters
        self.filter_name = filter_name
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_dict(cls, dict_representation: dict):
        return cls(
            dict_representation["filter_name"],
            *dict_representation["args"],
            subfilters=dict_representation["subfilters"],
            **dict_representation["kwargs"]
        )

    def get_dict_representation(self):
        dict_representation = super().get_dict_representation()
        dict_representation["is_filter"] = True
        dict_representation["filter_name"] = self.filter_name
        dict_representation["subfilters"] = [
            subfilter.get_dict_representation() for subfilter in self.subfilters
        ]
        dict_representation["args"] = self.args
        dict_representation["kwargs"] = self.kwargs
        return dict_representation

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
        return getattr(self, self._filter_prefix + self.filter_name)(
            subfiltered_features, *self.args, **self.kwargs
        )

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
