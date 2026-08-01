import unittest

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Staff, StaffPoint
from homr.staff_detection import (
    connect_staff_lines,
    recover_missing_staff_from_repeated_layout,
)


def makeBoundingBox(x: float, y: float) -> RotatedBoundingBox:
    w, h = 40.0, 2.0
    angle = 0.0
    return RotatedBoundingBox(((x, y), (w, h), angle), np.array([]))


def make_staff(center_y: float) -> Staff:
    return Staff(
        [
            StaffPoint(float(x), [center_y + offset for offset in (-20, -10, 0, 10, 20)], 0)
            for x in range(20, 201, 10)
        ]
    )


def draw_staff(staff: Staff, image: np.ndarray) -> None:
    for point in staff.grid:
        for y_value in point.y:
            image[int(round(y_value)), int(round(point.x))] = 1


class TestStaffDetection(unittest.TestCase):

    def test_connect_staff_lines(self) -> None:
        lines = [makeBoundingBox(100, 100), makeBoundingBox(50, 100), makeBoundingBox(150, 100)]
        result = connect_staff_lines(lines, 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].staff_fragments, [lines[1], lines[0], lines[2]])

    def test_recovers_one_pixel_backed_staff_from_repeated_paired_layout(self) -> None:
        all_staffs = [
            make_staff(center_y)
            for center_y in (100, 200, 380, 480, 660, 760, 940, 1040, 1220, 1320)
        ]
        missing_lower_staff = all_staffs.pop(5)
        staff_image = np.zeros((1400, 240), dtype=np.uint8)
        for staff in [*all_staffs, missing_lower_staff]:
            draw_staff(staff, staff_image)

        result = recover_missing_staff_from_repeated_layout(all_staffs, staff_image)

        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result[5].min_y, missing_lower_staff.min_y)
        self.assertAlmostEqual(result[5].max_y, missing_lower_staff.max_y)

    def test_does_not_recover_a_staff_without_segmented_line_support(self) -> None:
        all_staffs = [
            make_staff(center_y)
            for center_y in (100, 200, 380, 480, 660, 760, 940, 1040, 1220, 1320)
        ]
        all_staffs.pop(5)
        staff_image = np.zeros((1400, 240), dtype=np.uint8)
        for staff in all_staffs:
            draw_staff(staff, staff_image)

        result = recover_missing_staff_from_repeated_layout(all_staffs, staff_image)

        self.assertEqual(result, all_staffs)
