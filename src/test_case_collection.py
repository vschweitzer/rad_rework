"""
Defines TestCaseCollection, which offers loading and selection methods.
"""


import json
import os
import csv
from typing import List, Optional


import storable
import test_case


class TestCaseCollection(storable.Storable):
    """
    A collection of test_case.TestCase instances offering loading and selection
    methods.
    """

    test_cases: List[test_case.TestCase]
    annotation_mode: str

    def __init__(self, test_cases: List[test_case.TestCase]) -> None:
        """
        test_cases is a list of TestCases.
        """
        super().__init__()
        self.test_cases = test_cases

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["test_cases"] = [
            tc.get_dict_representation() for tc in self.test_cases
        ]
        return dict_representation

    def convert_annotations(
        self,
        annotation_mode: str = "2D",
        conversion_options: dict = {},
    ):
        """
        annotation_mode determines how TestCase annotations are treated.
            - None: No transformations are applied.
            - 2D: TestCase annotations are converted to 2D using the largest
                  slice.
            - 3D: Not implemented.
        """
        if annotation_mode is None:
            return
        elif annotation_mode == "2D":
            transformed_array: List[test_case.TestCase] = []

            overwrite: bool = False
            use_existing: bool = True
            along_axes: Optional[list] = None

            if "overwrite" in conversion_options:
                overwrite = conversion_options["overwrite"]
            if "use_existing" in conversion_options:
                use_existing = conversion_options["use_existing"]
            if "along_axes" in conversion_options:
                along_axes = conversion_options["along_axes"]

            for tc in self.test_cases:
                test_case.TestCase.to_2D(
                    tc,
                    None,
                    overwrite=overwrite,
                    use_existing=use_existing,
                    along_axes=along_axes,
                )

        else:
            raise NotImplementedError(
                f'"{annotation_mode}" transformation is not available.'
            )

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        file_ending: str = ".nii.gz",
        skip_invalid: bool = True,
    ):
        """
        Load TestCaseCollection from csv file.
        Item separators are commas (,), line separators are newlines (\\n).

        Expected columns are scan file name,pcr[,nar]
        """
        test_cases: List[test_case.TestCase] = []
        base_dir: str = os.path.dirname(csv_path)
        with open(csv_path, "r") as csv_file:
            # sniffer: csv.Sniffer = csv.Sniffer()
            # dialect = sniffer.sniff(csv_file.read(4096))
            reader = csv.reader(csv_file)

            for line in reader:
                scan_name: str = line[0]
                pcr: bool = line[1].lower() == "pcr"
                nar: Optional[int] = None
                if len(line) > 2:
                    try:
                        nar = int(line[2])
                    except Exception as ex:
                        pass
                if not scan_name.endswith(file_ending):
                    scan_name += file_ending
                scan_name = os.path.join(base_dir, scan_name)
                print(f"{scan_name} - {pcr} - {nar}")
                try:
                    tc: test_case.TestCase = test_case.TestCase.from_scan_path(
                        scan_name, pcr, nar
                    )
                except Exception as ex:
                    if not skip_invalid:
                        raise ex

                test_cases.append(tc)

        return cls(test_cases)


if __name__ == "__main__":
    import pprint

    pp: pprint.PrettyPrinter = pprint.PrettyPrinter(indent=2)
    tcc: TestCaseCollection = TestCaseCollection.from_csv(
        "../Dataset_V2/images_clean_NAR.csv"
    )
    pp.pprint(tcc.get_dict_representation())
    tcc.convert_annotations(conversion_options={"along_axes": [2]})
    pp.pprint(tcc.get_dict_representation())
