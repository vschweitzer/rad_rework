import unittest
from typing import Dict, List, Any

import import_parent
import feature_filter


class FeatureFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_features = []
        dummy_dict = {}
        entries: int = 20
        cases: int = 10
        for index in range(entries):
            dummy_dict[f"diagnostics_{index}"] = index / entries
            dummy_dict[str(index)] = index
        self.dummy_features = [dummy_dict] * cases

    def test_dict(self):
        ff: feature_filter.FeatureFilter = feature_filter.FeatureFilter(
            "do_nothing", 1, 2, 3, a=1, b=2, c=3
        )
        ff2: feature_filter.FeatureFilter = feature_filter.FeatureFilter.from_dict(
            ff.get_dict_representation()
        )

        self.assertEqual(ff.get_id(), ff2.get_id())

    # This filter wasn't available for testing, check if everything
    # works.
    def test_random_choice(self):
        filters: Dict[bool, Dict[bool, feature_filter.FeatureFilter]] = {}
        filtered_features: Dict[bool, Dict[bool, List[Dict[str, Any]]]] = {}
        fraction: float = 0.5
        starting_seed: int = 0
        for increase in [True, False]:
            filters[increase] = {}
            filtered_features[increase] = {}
            for invert in [True, False]:
                filters[increase][invert] = feature_filter.FeatureFilter(
                    "random_choice",
                    fraction,
                    random_seed=starting_seed,
                    increase_seed=increase,
                    invert=invert,
                    subfilters=[],
                )
                filtered_features[increase][invert] = filters[increase][invert].filter(
                    self.dummy_features
                )

        for invert in [True, False]:
            # Same seed should yield same selection
            self.assertEqual(
                filtered_features[True][invert], filtered_features[False][invert]
            )

        for increase in [True, False]:
            for invert in [True, False]:
                # increase_seed should lead to different seed after
                # execution
                self.assertNotEqual(
                    filters[increase][invert].kwargs["random_seed"] == starting_seed,
                    filters[increase][invert].kwargs["increase_seed"],
                )

                corrected_fraction: float = fraction if not invert else 1.0 - fraction
                total_features: int = len(self.dummy_features[0].keys())
                expected_feature_count: int = round(total_features * corrected_fraction)
                self.assertEqual(
                    len(filtered_features[increase][invert][0].keys()),
                    expected_feature_count,
                )

            reconstructed_features: List[Dict[str, Any]] = []
            for feature_sets in zip(
                filtered_features[increase][True], filtered_features[increase][False]
            ):
                reconstructed_features.append({**feature_sets[0], **feature_sets[1]})

            # Non-inverted + inverted should equal original
            self.assertEqual(reconstructed_features, self.dummy_features)


if __name__ == "__main__":
    unittest.main()
