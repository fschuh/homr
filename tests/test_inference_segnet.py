from __future__ import annotations

import numpy as np
import pytest

from homr.segmentation.inference_segnet import merge_patches

# Pytest assertions are intentional in this focused numerical test module.
# ruff: noqa: S101


def score_patch(labels: np.ndarray, class_count: int = 4) -> np.ndarray:
    scores = np.zeros((class_count, *labels.shape), dtype=np.float32)
    for class_index in range(class_count):
        scores[class_index] = (labels == class_index).astype(np.float32)
    return scores


def weighted_patch(class_scores: tuple[float, ...], height: int = 2, width: int = 2) -> np.ndarray:
    scores = np.zeros((len(class_scores), height, width), dtype=np.float32)
    for class_index, score in enumerate(class_scores):
        scores[class_index, :, :] = score
    return scores


def test_single_tile_reconstructs_unchanged() -> None:
    labels = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int64)

    reconstructed = merge_patches([score_patch(labels)], labels.shape, 3, 3)

    np.testing.assert_array_equal(reconstructed, labels)


def test_non_overlapping_tiles_reconstruct_unchanged() -> None:
    left = np.array([[1, 1], [2, 2]], dtype=np.int64)
    right = np.array([[3, 3], [0, 0]], dtype=np.int64)

    reconstructed = merge_patches(
        [score_patch(left), score_patch(right)], (2, 4), win_size=2, step_size=2
    )

    np.testing.assert_array_equal(reconstructed, np.hstack((left, right)))


def test_identical_overlapping_scores_are_unchanged() -> None:
    first = score_patch(np.ones((2, 2), dtype=np.int64))
    second = first.copy()

    reconstructed = merge_patches(
        [first, second, second, first, second, second], (1, 3), win_size=2, step_size=1
    )

    np.testing.assert_array_equal(reconstructed, np.array([[1, 1, 1]]))


def test_conflicting_overlap_chooses_highest_averaged_class_score() -> None:
    first = weighted_patch((0.0, 0.6, 0.4))
    second = weighted_patch((0.0, 0.1, 0.9))

    reconstructed = merge_patches(
        [first, second, second, first, second, second], (1, 3), win_size=2, step_size=1
    )

    np.testing.assert_array_equal(reconstructed, np.array([[1, 2, 2]]))


def test_overlap_cannot_invent_a_class_by_averaging_class_ids() -> None:
    first = score_patch(np.array([[1, 1], [1, 1]], dtype=np.int64), class_count=4)
    second = score_patch(np.array([[3, 3], [3, 3]], dtype=np.int64), class_count=4)

    reconstructed = merge_patches(
        [first, second, second, first, second, second], (1, 3), win_size=2, step_size=1
    )

    # The overlap is a tie between the two observed classes.  np.argmax's
    # stable tie break selects class 1; class 2 must never be synthesized.
    assert reconstructed[0, 1] in (1, 3)
    assert reconstructed[0, 1] != 2


def test_shifted_final_rows_and_columns_reconstruct_correctly() -> None:
    labels = np.array(
        [
            [0, 1, 2, 3, 0],
            [1, 2, 3, 0, 1],
            [2, 3, 0, 1, 2],
            [3, 0, 1, 2, 3],
            [0, 1, 2, 3, 0],
        ],
        dtype=np.int64,
    )
    patches = [score_patch(labels[y : y + 3, x : x + 3]) for y in (0, 2) for x in (0, 2)]

    reconstructed = merge_patches(patches, labels.shape, win_size=3, step_size=3)

    np.testing.assert_array_equal(reconstructed, labels)


def test_images_smaller_than_window_work() -> None:
    labels = np.array([[1, 2], [3, 0]], dtype=np.int64)

    reconstructed = merge_patches([score_patch(labels)], labels.shape, win_size=3, step_size=3)

    np.testing.assert_array_equal(reconstructed, labels)


def test_output_dimensions_exactly_match_input_page() -> None:
    image_shape = (7, 11)
    patch = score_patch(np.zeros((4, 4), dtype=np.int64))

    reconstructed = merge_patches([patch] * 6, image_shape, win_size=4, step_size=4)

    assert reconstructed.shape == image_shape


@pytest.mark.parametrize("step_size", [0, -1])
def test_merge_rejects_non_positive_step(step_size: int) -> None:
    with pytest.raises(ValueError, match="step_size"):
        merge_patches([score_patch(np.zeros((1, 1), dtype=np.int64))], (1, 1), 1, step_size)
