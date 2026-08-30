# homr

homr is an Optical Music Recognition (OMR) software designed to transform camera pictures of sheet music into
machine-readable MusicXML format. The resulting [MusicXML](https://www.w3.org/2021/06/musicxml40/) files can be further
processed using tools such as [musescore](https://musescore.com/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liebharc/homr/blob/main/colab.ipynb)

You might also want to check out [Andromr](https://github.com/aicelen/Andromr), an Android app for optical music recognition using homr.

## About this fork

This is a fork of [liebharc/homr](https://github.com/liebharc/homr), modified to
support **interactive sheet-music viewers** — applications that need to know
where on the page each recognized note actually is, so they can highlight it,
select it, and follow it during playback. Upstream homr produces MusicXML; this
fork additionally produces the geometry that ties that MusicXML back to the
image.

### Visual sidecar

Running with `--output-visual-sidecar` writes `<image>.homr.visual.json` beside
the MusicXML. It links **every pitched MusicXML note to exactly one notehead on
the page**, or explicitly marks it as unlinked — never ambiguously, and never
partially. A viewer can therefore highlight the correct notehead for any note it
plays without guessing.

The sidecar also reports each notehead's `staff_position` (its diatonic line or
space) measured from the printed staff-line geometry, *independently* of what
the transformer predicted the pitch to be. That gives consumers a way to
cross-check recognized pitch against what is actually printed.

The contract is strict by design: consumers must not infer pitch, repair links,
or synthesize missing noteheads. All of that work happens inside homr, where the
source image is still available.

### Recognition and geometry repairs

Making that one-to-one guarantee hold on real scores required fixing a number of
cases where noteheads could not be matched reliably:

- **Chords** — displaced (seconds), dense whole-note stacks, cross-staff chords,
  shared stems, and hollow noteheads split into fragments by segmentation
- **Staff geometry** — staff positions refit against printed staff-line pixels
  rather than a single possibly-corrupted grid sample, which fixes false pitch
  mismatches on skewed scans
- **Grand staves** — recovery of a missing staff in repeated grand-staff
  layouts, which previously caused bass-staff notes to be dropped before
  transformer inference
- **Accidentals** — resolved accidentals preserved in sidecar pitches, so A♭3 is
  no longer reported as A3

### Improved segmentation accuracy

This one affects recognition quality for everyone, sidecar or not. SegNet runs
on 320×320 tiles; the upstream merger picked a class *within* each tile and then
averaged the resulting integer class labels across the page. Averaging class IDs
is not a meaningful operation — the average of classes 1 and 3 is 2, a class
neither tile predicted — and it discarded the per-class evidence exactly where
tiles disagree, at their boundaries.

The corrected merger accumulates full per-class score maps in page coordinates
and applies `argmax` once, at the end. On a 31-page pitch-reference corpus with
identical models and settings, total errors dropped from 8 to 5 and exactly
matched pages rose from 27 to 28, **with no page regressing**. See
`docs/source/recognition_findings.rst` for the full method and provenance.

### Evaluation tooling

A new `homr-visual-eval` command checks a sidecar against the v3 contract. It
can evaluate already-generated artifacts without re-running inference, which
makes it practical to validate a whole corpus after a change.

### Upstream

These changes are not endorsed by the upstream authors. This fork diverged from
upstream at commit `8b5dcf7` and was modified between June and August 2026; the
git history has the full record.

## Prerequisites

- Python 3.11
- Poetry
- Optional: NVidia GPU with CUDA 12.1

## Getting started (uv)

The easiest way to get started is using `uvx` (`uv` must be installed). Note that is does not make use of the GPU.
- `uvx homr <img>`
- The resulting MusicXML file will be saved in the same directory as the input image
- To combine the MusicXML results from multiple images, you can use [relieur](https://github.com/papoteur-mga/relieur)

## Getting started (poetry)

- Clone the repository
- Install dependencies for:
  - GPU (requires CUDA): `poetry install --only main,gpu`
  - CPU: `poetry install --only main`
  - Development: `poetry install`
- Run the program using `poetry run homr <image>`
- The resulting MusicXML file will be saved in the same directory as the input image
- To combine the MusicXML results from multiple images, you can use [relieur](https://github.com/papoteur-mga/relieur)

## Example

The example below provides an overview of the current performance of the implementation. While some errors are present
in the output, the overall structure remains accurate.

|                                          Original Image                                           |                                                                               homr Result                                                                                |
| :-----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/BreezeWhite/oemer/blob/main/figures/tabi.jpg?raw=true" width="400" > | <img src="https://github.com/liebharc/homr/blob/main/figures/tabi.svg?raw=true" alt="Go to https://github.com/liebharc/homr if this image isn't displayed" width="400" > |

The homr result is obtained by processing the [homr output](figures/tabi.musicxml) and rendering it
with [musescore](https://musescore.com/).

## Limitations

The current implementation focuses on pitch and rhythm information on the bass or treble clef, neglecting dynamics,
articulation, double sharps/flats, and other musical symbols.

## Technical Details

homr uses a two-stage pipeline: **segmentation** for structural analysis followed by **semantic symbol recognition** via transformer models.

### Stage 1: Image Segmentation and Structural Analysis

homr employs UNet-based segmentation models (adapted from [oemer](https://github.com/BreezeWhite/oemer)) to extract structural components from the sheet music image:

- **Staff lines and symbols**: Detected via trained segmentation networks that identify:
  - Staff line fragments
  - Note heads
  - Stems and rests
  - Bar lines
  - Clefs and key signatures

The segmentation process generates bounding boxes for each detected element. These predictions serve as inputs for the staff detection algorithm.

### Stage 2: Staff Detection and Merging

Using the segmentation outputs, homr constructs staffs through the following steps:

1. **Staff Anchor Detection**: The algorithm identifies "staff anchors" (clefs and bar lines) that serve as reference points for accurate staff localization, even when symbols partially obscure staff lines.

2. **Unit Size Estimation**: For each staff, the algorithm calculates the "unit size" (distance between staff lines). This accommodates camera perspective variations and non-uniform staff spacing.

3. **Staff Reconstruction**: Around each anchor, five staff lines are located and the remaining staff structure is reconstructed using the estimated unit size.

4. **Grand Staff Merging**: Braces and brackets are identified to merge related staffs, supporting:
   - Grand staffs (piano, organ)
   - Multiple voices on a single staff
   - Mixed instrument groups

### Stage 3: Semantic Symbol Recognition via Transformer

Each staff is dewarped (perspective-corrected) and passed through a transformer-based model (based on [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR)) that performs **end-to-end symbol sequence recognition**. The model outputs:

- **Rhythm symbols**: Note durations, rests, and tuplet information
- **Pitch information**: Absolute pitch values with accidentals (sharps, flats, naturals)
- **Articulation marks**: Accents, staccato, tenuto, and slur markers
- **Performance annotations**: Dynamic expressions and other musical notation

The transformer model generates these predictions in sequence, processing the dewarped staff image to understand the spatial and temporal relationships between musical symbols.

**Note**: The transformer output provides the sequence of symbols but does not include explicit positional information (horizontal or vertical coordinates). However, the model computes the center of attention as a byproduct of the attention mechanism, which can be used to estimate the focus point on the staff image.

### Stage 4: MusicXML Output

The symbol sequence is converted into MusicXML format and saved to disk. The resulting file can be processed with tools like [musescore](https://musescore.com/) or [relieur](https://github.com/papoteur-mga/relieur) (for multi-image combinations).

## Citation

If you use this code in your research work, please cite [oemer](https://github.com/BreezeWhite/oemer)
and [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR).

## Name

The name "homr" stands for Homer's Optical Music Recognition (OMR), leaving the interpretation of "Homer" to the user's
discretion, whether referring to the ancient poet [Homer](https://en.wikipedia.org/wiki/Homer) or the iconic character
from [The Simpsons](https://en.wikipedia.org/wiki/The_Simpsons).

## Thanks

This project builds upon previous work, including:

- The segmentation models of [oemer](https://github.com/BreezeWhite/oemer)
- The transformer model of [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR)
- The starter template provided by [Benjamin Roland](https://github.com/Parici75/python-poetry-bootstrap)
