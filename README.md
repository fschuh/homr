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

The sidecar also reports, for every rest in the MusicXML, whether a printed rest
backs it. Segmentation has no rest class, so homr looks for free-standing ink of a
rest's shape where the transformer placed the rest. Rests without such ink, and
notes read without a pitch that MusicXML would otherwise write as rests, are marked
unsupported, so a viewer can show them as rests the page does not print.

It likewise reads each linked note's value from the page — a hollow or filled
notehead, a stem, the flags or beams at the stem's end, an augmentation dot — and
reports where that printed value disagrees with the recognized one. The reader
reports nothing where any of its checks is in doubt, such as two voices sharing a
notehead. Like the rest report, it never changes the MusicXML.

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

### Note timing in MusicXML

This also affects everyone. The transformer's tokens say which notes start
together, not when. The MusicXML writer started each group after the shortest
note it had just written, ignoring notes the other hand still held, so wherever
one hand moved under a held note every later note started late: staves got gaps
that score editors fill with rests the score does not have, and bars overflowed.
Groups now start when the earliest sounding note ends.

The same measurement now decides the time signature's beat count, separately for
each meter section, without counting a staff whose only content in a bar is a
rest. Such a rest is written as a measure rest filling the bar, whatever glyph
was read, and symbols the transformer chords with notes, such as a barline, are
no longer written as rests. On a 240-page corpus, bars with a staff gap or
overflow fell from 1,740 to 1,371 of 3,865, with no page regressing.

Starting groups when the earliest note ends trusts every length, though. Where one
note is read shorter than printed while the other hand holds a note, such as a bass
16th read as a 32nd under a whole-note chord, the next group starts off the beat.
From there each group starts at whichever of two ends comes first: the rest of the
bar rushes, and the staff falls silent before the barline. The writer now places
every bar both ways and keeps the earliest-end placement unless starting each group
after the shortest note of the group before leaves every staff at least as close to
the bar's length and one staff closer. Then the misread only moves the notes after it
a little, and they keep their spacing. No note changes its value, and a bar with a
shared-notehead repair (below) keeps the earliest-end placement.

On the corpus, with the visual sidecar, this places 78 bars of 18 pieces after the
shortest note and changes no other bar. Bars where a staff stops at least an eighth
before the barline, and an eighth earlier than under the old rule alone, fell from 94
to 31. Bars with every staff within an eighth of the barline rose from 3,072 to 3,116
of 3,816, and no bar that ended exactly on it changed. Of the changed bars, 31 are in
pieces with MIDI exports: in 26 the notes' offsets from the MIDI spread less within
the bar, and the other five, checked note by note against the MIDI or the page, keep
the printed spacing where the earliest end rushed them. The changed bars match fewer
MIDI onsets exactly, because the notes after a misread now start a little late
instead of early and crowded.

### Notes after a shared notehead

Two voices that sound one pitch at the same moment often share a printed
notehead: an eighth note on the first note of a 16th run, or a held melody note
on the first note of an arpeggio. The transformer reads such a head once,
usually with the longer value, so the notes after it started late and the bar
overflowed; in Chopin's Op. 25 No. 9, 38 of 51 bars did.

With `--output-visual-sidecar`, homr now reads both stems of such a head on the
page. Where one carries a shorter value than the token, the notes after the head
start when that shorter value ends, as printed. No note is added and none changes
its value. A bar's repairs are applied together, and only when the bar overflows
or falls short as recognized and adds up exactly to its time signature with
them. Tuplets are not read, so where the shorter note is an unmarked triplet the
notes after it start after a plain eighth, as the transformer reads the rest of
that run. Each shared notehead and what became of it is recorded in the sidecar.
`--no-timing-repairs` turns this off.

On the 240-page corpus this changed 10 pages of 5 pieces. The 119 repairs were
each checked against the page: every one sits on a shared notehead, and the next
notes now start after the value printed on its shorter stem. Bars in which a
staff runs past the time signature fell from 654 to 600, and no page gained a
bar that runs over or falls short. On 131 engraved pages with MIDI exports of
their source files, the 5 affected bars now start every note where the MIDI
does, from 14–42% of notes before, and no note in any other bar moved.

### Printed metronome marks

The transformer reads the staff, not the text above it, so homr's MusicXML had no
tempo and players fell back to their own default, usually quarter = 120. That
distorted more than the speed. Lucca's Theme from Chrono Trigger prints dotted
quarter = 140 for its 6/8 bars and quarter = 140 where it changes to 2/4, keeping
the beat steady; at a fixed quarter rate its 6/8 bars played with a beat a third
slower than its 2/4 bars.

With `--output-visual-sidecar`, homr now reads the metronome marks printed above
each system, such as "♩ = 120", "Allegro (♩ = 120)" or "♩. = 140", and writes
each as a metronome with a sound tempo in quarters per minute at the start of the
measure its text begins over; text that begins between two measures belongs to
the next one. The note is read from its shape on the page (hollow or filled
head, stem, flags, dot) and only the digits go to OCR. A mark whose note cannot
be read is left out, so that part of the page has no tempo, as before. A range
such as "64-68" plays at its first number, as notation programs play it. Marks
without a number ("Allegro", "Met - 96") and equivalences without one
("♩. = ♩") are not read. A tempo given with `--output-metronome` wins over the
printed ones. Each mark and what became of it is recorded in the sidecar.

On the 240-page corpus and 28 pages of 19 further pieces, 96 marks were written
on 91 pages of 84 pieces, and each was checked against the page for its note,
its number and its measure. Two marks of an old engraving were left unread, and
one restated tempo was not written because its measure already had a mark.
Nothing else in the MusicXML changed on any page. For 74 of the 80 tempos
written in pieces with MIDI exports of their source files, the MIDI holds the
same tempo; the other 6 match their printed marks where the MIDI plays another
tempo, such as Super Mario Bros' 1-Up, which prints quarter = 225 and plays at 100.

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
