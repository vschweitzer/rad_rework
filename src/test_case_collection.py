"""
Defines TestCaseCollection, which offers loading and selection methods.
"""


import json
import os
import csv
from typing import List


import storable
import test_case


class TestCaseCollection(storable.Storable):
    """
    A collection of test_case.TestCase instances offering loading and selection
    methods.
    """

    test_cases: List[test_case.TestCase]

    def __init__(self, test_cases: List[test_case.TestCase]) -> None:
        super().__init__()
        self.test_cases = test_cases

    def get_dict_representation(self):
        dict_representation: dict = super().get_dict_representation()
        dict_representation["test_cases"] = [
            tc.get_dict_representation() for tc in self.test_cases
        ]
        return dict_representation

    def __str__(self) -> str:
        return json.dumps(self.get_dict_representation())

    @classmethod
    def from_csv(cls, csv_path: str, file_ending: str = ".nii.gz"):
        """
        Load TestCaseCollection from csv file.
        Item separators are commas (,), line separators are newlines (\\n).
        """
        test_cases: List[test_case.TestCase] = []
        base_dir: str = os.path.dirname(csv_path)
        with open(csv_path, "r") as csv_file:
            # sniffer: csv.Sniffer = csv.Sniffer()
            # dialect = sniffer.sniff(csv_file.read(4096))
            reader = csv.reader(csv_file)

            for line in reader:
                scan_name: str = line[0]
                pcr_status: bool = line[1].lower() == "pcr"
                if not scan_name.endswith(file_ending):
                    scan_name += file_ending
                scan_name = os.path.join(base_dir, scan_name)
                print(f"{scan_name} - {pcr_status}")
                test_cases.append(
                    test_case.TestCase.from_scan_path(scan_name, int(pcr_status))
                )
        return cls(test_cases)


if __name__ == "__main__":
    tcc: TestCaseCollection = TestCaseCollection.from_csv(
        "../Dataset_V2/images_clean.csv"
    )
    print(tcc)
