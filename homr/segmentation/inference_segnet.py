import hashlib
import lzma
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from homr.onnx_providers import (
    coreml_available,
    coreml_mlprogram_providers,
    cuda_available,
)
from homr.segmentation.config import (
    segmentation_version,
    segnet_path_onnx,
    segnet_path_onnx_fp16,
)
from homr.simple_logging import eprint
from homr.type_definitions import NDArray

SEGNET_WINDOW_SIZE = 320
MERGE_STRATEGY = "mean-class-scores"


class Segnet:
    def __init__(self, use_gpu_inference: bool) -> None:
        self.use_gpu = False
        if use_gpu_inference and cuda_available():
            try:
                # I had this issue: https://github.com/microsoft/onnxruntime/issues/21684
                # If torch is installed, this fixes
                # "libcudnn.so.9: cannot open shared object file"
                ort.preload_dlls()
                self.model = ort.InferenceSession(
                    segnet_path_onnx_fp16, providers=["CUDAExecutionProvider"]
                )
                self.fp16 = True
                # Segnet always binds its output on the CPU, so use_gpu only needs to track
                # that a GPU provider is active (used for logging/diagnostics).
                self.use_gpu = True
            except Exception as e:
                eprint(
                    "Error while trying to load model on the GPU. You probably don't have a compatible gpu"  # noqa: E501
                )
                eprint(e)
                self.model = ort.InferenceSession(segnet_path_onnx_fp16)
                self.fp16 = True
        elif use_gpu_inference and coreml_available():
            try:
                # CPUAndGPU skips the (slow, CPU-bound) ANE specialization that
                # "ALL" performs at session creation and on the first inference.
                # Steady-state batches are slightly slower than with "ALL", but
                # a session only lives for one image, so the warmup dominates
                # (measured on an M1: ~1.9 s vs ~2.6 s for a 4-staff page).
                self.model = ort.InferenceSession(
                    segnet_path_onnx_fp16,
                    providers=coreml_mlprogram_providers(
                        segnet_path_onnx_fp16, compute_units="CPUAndGPU"
                    ),
                )
                self.fp16 = True
                self.use_gpu = True
            except Exception as e:
                eprint("Error while trying to load model on CoreML, falling back to the CPU")
                eprint(e)
                if os.path.exists(segnet_path_onnx):
                    self.model = ort.InferenceSession(segnet_path_onnx)
                    self.fp16 = False
                else:
                    self.model = ort.InferenceSession(segnet_path_onnx_fp16)
                    self.fp16 = True
        elif use_gpu_inference:
            # --gpu force without a GPU execution provider: only the fp16 model
            # has been downloaded, run it on the CPU.
            eprint("No GPU execution provider available, running on the CPU instead")
            self.model = ort.InferenceSession(segnet_path_onnx_fp16)
            self.fp16 = True
        else:
            self.model = ort.InferenceSession(segnet_path_onnx)
            self.fp16 = False

        self.io_binding = self.model.io_binding()
        self.device_id = 0

        self.input_name = self.model.get_inputs()[0].name  # size: [batch_size, 3, 320, 320]
        self.output_name = self.model.get_outputs()[0].name

    def run(self, input_data: NDArray) -> NDArray:
        if self.fp16:
            self.io_binding.bind_cpu_input("input", input_data.astype(np.float16))
        else:
            self.io_binding.bind_cpu_input("input", input_data.astype(np.float32))

        self.io_binding.bind_output("output", "cpu")
        self.model.run_with_iobinding(self.io_binding)
        out = self.io_binding.get_outputs()[0].numpy()
        return out


class ExtractResult:
    def __init__(
        self,
        filename: Path,
        original: NDArray,
        staff: NDArray,
        symbols: NDArray,
        stems_rests: NDArray,
        notehead: NDArray,
        clefs_keys: NDArray,
    ):
        self.filename = filename
        self.original = original
        self.staff = staff
        self.symbols = symbols
        self.stems_rests = stems_rests
        self.notehead = notehead
        self.clefs_keys = clefs_keys


def extract_patch(image: NDArray, y: int, x: int, win_size: int) -> NDArray:
    """
    Returns a full-size (3, win_size, win_size) patch.
    Pads with white pixes if the patch exceeds image boundaries.
    """
    c, h, w = image.shape
    patch = np.full((c, win_size, win_size), 255, dtype=image.dtype)

    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + win_size, h)
    x1 = min(x + win_size, w)

    py0 = 0
    px0 = 0
    py1 = py0 + (y1 - y0)
    px1 = px0 + (x1 - x0)

    patch[:, py0:py1, px0:px1] = image[:, y0:y1, x0:x1]
    return patch


def merge_patches(
    patches: list[NDArray], image_shape: tuple[int, int], win_size: int, step_size: int
) -> NDArray:
    """Merge SegNet score patches and choose a class for each page pixel.

    SegNet returns one score map per class.  Keeping those maps until after
    all tiles have been merged is important: averaging integer class labels
    can produce a value that is not a class at all (for example, averaging
    classes 1 and 3 produces 2).  The overlap coverage is uniform for now;
    the caller supplies the tile step separately so overlap can be evaluated
    without changing this merge rule.
    """
    if not patches:
        raise ValueError("At least one SegNet patch is required")
    if step_size <= 0:
        raise ValueError("step_size must be greater than zero")
    if win_size <= 0:
        raise ValueError("win_size must be greater than zero")

    first_patch = np.asarray(patches[0])
    if first_patch.ndim != 3:
        raise ValueError("SegNet patches must have shape (classes, height, width)")
    class_count = first_patch.shape[0]
    if class_count == 0:
        raise ValueError("SegNet patches must contain at least one class")

    score_sum = np.zeros((class_count, *image_shape), dtype=np.float32)
    coverage = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    # Keep this grid identical to inference(), including the extra starts
    # needed when a page dimension is smaller than the model window.
    for iy in range(0, max(image_shape[0], win_size), step_size):
        y = min(iy, image_shape[0] - win_size)
        y0 = max(y, 0)
        y1 = min(y + win_size, image_shape[0])

        for ix in range(0, max(image_shape[1], win_size), step_size):
            x = min(ix, image_shape[1] - win_size)
            x0 = max(x, 0)
            x1 = min(x + win_size, image_shape[1])

            patch = patches[idx]
            if patch.shape != first_patch.shape:
                raise ValueError("All SegNet patches must have the same shape")
            ph = y1 - y0
            pw = x1 - x0

            score_sum[:, y0:y1, x0:x1] += np.asarray(patch[:, :ph, :pw], dtype=np.float32)
            coverage[y0:y1, x0:x1] += 1
            idx += 1

    if idx != len(patches):
        raise ValueError("More SegNet patches were supplied than the page requires")

    # Avoid division by zero for malformed/empty image shapes while keeping
    # normal page pixels as the uniform mean of their contributing tiles.
    coverage[coverage == 0] = 1
    score_sum /= coverage[None, :, :]

    return np.argmax(score_sum, axis=0).astype(np.int64)


def inference(
    image_org: NDArray, use_gpu_inference: bool, batch_size: int, step_size: int, win_size: int
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Inference function for the segementation model.
    Args:
        image_org(NDArray): Array of the input image
        batch_size(int): Mainly for speeding up GPU performance. Minimal impact on CPU speed.
        step_size(int): How far the window moves between to input images.
        win_size(int): Debug only.

    Returns:
        ExtractResult class.
    """
    eprint("Starting Inference.")
    t0 = perf_counter()
    if step_size < 0:
        step_size = win_size // 2
    if not 1 <= step_size <= win_size:
        raise ValueError(f"step_size must be between 1 and {win_size}")

    model = Segnet(use_gpu_inference)

    image_org = cv2.cvtColor(image_org, cv2.COLOR_GRAY2BGR)
    image = np.transpose(image_org, (2, 0, 1)).astype(np.float32)

    c, h, w = image.shape
    data: list[NDArray] = []
    batch: list[NDArray] = []

    for y_loop in range(0, max(h, win_size), step_size):
        y = min(y_loop, h - win_size)
        for x_loop in range(0, max(w, win_size), step_size):
            x = min(x_loop, w - win_size)

            hop = extract_patch(image, y, x, win_size)

            batch.append(hop)

            if len(batch) == batch_size:
                batch_out = model.run(np.stack(batch, axis=0))
                for out in batch_out:
                    data.append(out)
                batch.clear()

    if batch:
        batch_out = model.run(np.stack(batch, axis=0))
        for out in batch_out:
            data.append(out)

    eprint(f"Segnet Inference time: {perf_counter() - t0}; batch_size {batch_size}")

    merged = merge_patches(
        data, (int(image_org.shape[0]), int(image_org.shape[1])), win_size, step_size
    )

    stems_rests = (merged == 1).astype(np.uint8)
    notehead = (merged == 2).astype(np.uint8)
    clefs_keys = (merged == 3).astype(np.uint8)
    staff = (merged == 4).astype(np.uint8)
    symbols = (merged == 5).astype(np.uint8)

    return staff, symbols, stems_rests, notehead, clefs_keys


def extract(
    original_image: NDArray,
    img_path_str: str,
    use_cache: bool = False,
    use_gpu_inference: bool = True,
    batch_size: int = 8,
    step_size: int = SEGNET_WINDOW_SIZE,
    win_size: int = SEGNET_WINDOW_SIZE,
) -> ExtractResult:
    if step_size < 0:
        step_size = win_size // 2
    if not 1 <= step_size <= win_size:
        raise ValueError(f"step_size must be between 1 and {win_size}")

    img_path = Path(img_path_str)
    f_name = os.path.splitext(img_path.name)[0]
    npy_path = img_path.parent / f"{f_name}.npy"
    loaded_from_cache = False
    if npy_path.exists() and use_cache:
        eprint("Found a cache")
        file_hash = hashlib.sha256(original_image).hexdigest()  # type: ignore
        with lzma.open(npy_path, "rb") as f:
            staff = np.load(f)
            notehead = np.load(f)
            symbols = np.load(f)
            stems_rests = np.load(f)
            clefs_keys = np.load(f)
            cached_file_hash = f.readline().decode().strip()
            model_name = f.readline().decode().strip()
            cached_step_size = f.readline().decode().strip()
            cached_merge_strategy = f.readline().decode().strip()
            if (
                cached_file_hash == ""
                or model_name == ""
                or cached_step_size == ""
                or cached_merge_strategy == ""
            ):
                eprint("Cache is missing meta information, skipping cache")
            elif file_hash != cached_file_hash:
                eprint("File hash mismatch, skipping cache")
            elif model_name != segmentation_version:
                eprint("Models have been updated, skipping cache")
            elif cached_step_size != str(step_size):
                eprint("Segnet step size changed, skipping cache")
            elif cached_merge_strategy != MERGE_STRATEGY:
                eprint("Segnet merge strategy changed, skipping cache")
            else:
                loaded_from_cache = True
                eprint("Loading from cache")

    if not loaded_from_cache:
        staff, symbols, stems_rests, notehead, clefs_keys = inference(
            original_image,
            use_gpu_inference=use_gpu_inference,
            batch_size=batch_size,
            step_size=step_size,
            win_size=win_size,
        )
        if use_cache:
            eprint("Saving cache")
            file_hash = hashlib.sha256(original_image).hexdigest()  # type: ignore
            with lzma.open(npy_path, "wb") as f:
                np.save(f, staff)
                np.save(f, notehead)
                np.save(f, symbols)
                np.save(f, stems_rests)
                np.save(f, clefs_keys)
                f.write((file_hash + "\n").encode())
                f.write((segmentation_version + "\n").encode())
                f.write((str(step_size) + "\n").encode())
                f.write((MERGE_STRATEGY + "\n").encode())

    original_image = cv2.resize(original_image, (staff.shape[1], staff.shape[0]))

    return ExtractResult(
        img_path, original_image, staff, symbols, stems_rests, notehead, clefs_keys
    )
