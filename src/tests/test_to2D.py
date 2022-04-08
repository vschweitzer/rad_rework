import copy

import import_parent
import test_case

if __name__ == "__main__":
    test_scan_path: str = "../Dataset_V2/MR86.nii.gz"
    tc: test_case.TestCase = test_case.TestCase.from_scan_path(
        scan_path=test_scan_path, pcr=True, nar=1
    )
    axes: list = list(range(3))
    for axis in axes:
        to_convert: test_case.TestCase = copy.deepcopy(tc)
        to_convert.to_2D(overwrite=True, use_existing=False, along_axes=[axis])
        print(
            f"Slice along axis: Expected: {axis}, Received: {to_convert.anno_file.get_slice_dimension(to_convert.anno_file.image)}"
        )
