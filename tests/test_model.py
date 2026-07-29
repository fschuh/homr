import unittest

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import MultiStaff, Staff, StaffPoint


def make_staff(number: int) -> Staff:
    y_points = [10 * i + 100 * float(number) for i in range(5)]
    return Staff([StaffPoint(0.0, y_points, 0)])


def make_staff_grid(
    number: int,
    x_positions: list[int],
    y_curve: float = 0.0,
) -> Staff:
    return Staff(
        [
            StaffPoint(
                float(x),
                [10 * i + 100 * float(number) + y_curve * x for i in range(5)],
                y_curve,
            )
            for x in x_positions
        ]
    )


def make_connection(number: int) -> RotatedBoundingBox:
    rect = ((float(number), float(number)), (number, number), float(number))
    contours = np.empty((0, 0))
    return RotatedBoundingBox(rect, contours)


def make_brace(
    upper_staff_index: int,
    lower_staff_index: int,
    x_bias: float = 0.0,
    y_bias: float = 0.0,
) -> RotatedBoundingBox:
    center_x = 0.0
    min_y = upper_staff_index * 100
    max_y = (
        lower_staff_index * 100 + 40
    )  # because one staff is 40px high and starts from n*100px, see def make_staff()
    center_y = (min_y + max_y) / 2

    center_x += x_bias
    center_y += y_bias

    width = 10.0
    height = max_y - min_y
    rect = ((center_x, center_y), (width, height), 0.0)
    contours = np.empty((0, 0))
    return RotatedBoundingBox(rect, contours)


class TestModel(unittest.TestCase):

    def test_staff_merge_recovers_truncated_grandstaff_span(self) -> None:
        upper = make_staff_grid(0, [0, 10, 20], y_curve=0.1)
        lower = make_staff_grid(1, [0, 10, 20, 30, 40, 50], y_curve=0.1)

        result = upper.merge(lower)

        self.assertTrue(result.is_grandstaff)
        self.assertEqual([point.x for point in result.grid], [0, 10, 20, 30, 40, 50])
        self.assertEqual(result.max_x, 50)
        self.assertEqual(len(result.grid[-1].y), 10)
        np.testing.assert_allclose(
            result.grid[-1].y,
            [5, 15, 25, 35, 45, 105, 115, 125, 135, 145],
        )

    def test_staff_merge_does_not_extrapolate_from_too_little_overlap(self) -> None:
        upper = make_staff_grid(0, [0, 10, 20])
        lower = make_staff_grid(1, [10, 20, 30])

        result = upper.merge(lower)

        self.assertEqual([point.x for point in result.grid], [10, 20])

    def test_multi_staff_merge(self) -> None:
        staff1 = MultiStaff(
            [make_staff(1), make_staff(2)], [make_connection(1), make_connection(2)]
        )
        staff2 = MultiStaff(
            [staff1.staffs[1], make_staff(3)], [staff1.connections[1], make_connection(3)]
        )
        result1 = staff1.merge(staff2)
        self.assertEqual(result1.staffs, [staff1.staffs[0], staff1.staffs[1], staff2.staffs[1]])
        self.assertEqual(
            result1.connections,
            [staff1.connections[0], staff1.connections[1], staff2.connections[1]],
        )

        result2 = staff2.merge(staff1)
        self.assertEqual(result2.staffs, [staff1.staffs[0], staff2.staffs[0], staff2.staffs[1]])
        self.assertEqual(
            result2.connections,
            [staff2.connections[0], staff2.connections[1], staff1.connections[0]],
        )

    def test_create_grandstaffs_empty(self) -> None:
        ms = MultiStaff([], [])

        result = ms.create_grandstaffs([make_brace(0, 1)])

        self.assertIs(result, ms)

    def test_create_grandstaffs(self) -> None:
        ms = MultiStaff([make_staff(i) for i in range(4)], [])

        result = ms._select_grandstaffs([make_brace(0, 1), make_brace(2, 3)])

        self.assertEqual([pair.get_index() for pair in result], [[0, 1], [2, 3]])

    def test_create_grandstaffs_overlapping(self) -> None:
        ms = MultiStaff([make_staff(i) for i in range(4)], [])

        result = ms._select_grandstaffs([make_brace(0, 1, x_bias=20.0), make_brace(1, 2)])

        self.assertEqual([pair.get_index() for pair in result], [[1, 2]])

    def test_create_grandstaffs_invalid_brace(self) -> None:
        ms = MultiStaff([make_staff(i) for i in range(3)], [])

        result = ms._select_grandstaffs([make_brace(0, 1, x_bias=500.0)])

        self.assertEqual(result, [])
