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
together, not when. The MusicXML writer starts each group after the shortest note
it has just written. Wherever one hand moves under a note the other hand still
holds, the notes after it start late and the bar overflows: one late hiccup, after
which the notes keep their spacing.

Upstream homr starts each group when the earliest sounding note ends instead,
which puts such hands on their beats but trusts every length. Where one note is
read shorter than printed while the other hand holds a note, such as a bass 16th
read as a 32nd under a whole-note chord, the next group starts off the beat. From
there each group starts at whichever of two ends comes first: the rest of the bar
rushes, and the staff falls silent before the barline. Pages read right rush too,
where a staff holds three voices and an inner one ends before the melody does, as
in Chopin's Op. 25 No. 3. So this fork places a bar by the earliest end only where
the bar proves it: its meter is trusted, read on the page or carried from the
previous page (see below), and that placement ends every staff exactly on the
barline where starting after the shortest note does not. Every other bar is placed
after the shortest note. A guessed meter proves nothing, so without
`--read-time-signatures` every bar is placed that way. No note changes its value.

The time signature's beat count, when it is guessed, comes from the typical bar
length, measured by the earliest end and separately for each meter section, without
counting a staff whose only content in a bar is a rest. Such a rest is written as a
measure rest filling the bar, whatever glyph was read. Only in a bar shorter than
the meter, such as a pickup, does a rest that lasts as read exactly as long as the
other staff keep its value. Symbols the transformer chords with notes, such as a
barline, are no longer written as rests.

Measured against MuseScore scores of 26 pieces (1,943 bars; see below), bars as
long as the score's rose from 1,192 to 1,230, and to 1,423 with
`--read-time-signatures --repair-tuplets`. On the 240-page corpus, the pieces with
such a score (1,548 bars) went from 894 to 962 and 1,080. No bar on the MuseScore
pages ends earlier than before; with the options, one note starts a 32nd early,
where a double-dotted note is read as dotted, in a bar that still ends on its
barline. On the corpus two bars end earlier. In one the comparison errs: aligned by
pitch, a repeated figure matches the bar before, and the bar is right. In the other,
two bars merged by a lost barline no longer stall five beats on a rest read where
none is printed, and end half a beat short where two eighths are read as triplets.
Three bars end later than before (one with the options): two where a staff holding
only a rest fills a guessed meter longer than the bar, and one where a chord the
transformer merged is split again. On the rest of the corpus and on 131 video-game
pages, every bar that starts a note earlier than before ends exactly on the meter
printed for it, here or on an earlier page, except one bar on a page where none is
read, which now fills its guessed 6/8 evenly.

### Notes after a shared notehead

Two voices that sound one pitch at the same moment often share a printed
notehead: an eighth note on the first note of a 16th run, or a held melody note
on the first note of an arpeggio. The transformer reads such a head once,
usually with the longer value, so the notes after it started late and the bar
overflowed; in Chopin's Op. 25 No. 9, 38 of 51 bars did.

With `--output-visual-sidecar`, homr now reads both stems of such a head on the
page. Where one carries a shorter value than the token, the notes after the head
start when that shorter value ends, as printed. No note is added and none changes
its value. A bar's repairs are applied together, and only when the bar, placed as
it is written (see above), overflows or falls short as recognized and adds up
exactly to its time signature with them. Tuplets are not read, so where the shorter note is an unmarked triplet the
notes after it start after a plain eighth, as the transformer reads the rest of
that run. Each shared notehead and what became of it is recorded in the sidecar.
`--no-timing-repairs` turns this off.

On the 240-page corpus this changes 13 pages of 8 pieces. The 124 repairs were
each checked against the page: every one sits on a shared notehead, and in all but
one the next notes now start after the value printed on its shorter stem. That one
reads a triplet eighth's stem as a 16th, and its bar's bass ends a 16th early. Bars
in which a staff runs past the time signature fall from 1,022 to 968 of 3,885, and
no other page gains a bar that runs over or falls short. On 131 engraved pages with
MIDI exports of their source files, the 5 affected bars now start every note where
the MIDI does, from 14–42% of notes before, and no note in any other bar moved.

### Printed time signatures and tuplets

On classical pages the largest duration error is a missed tuplet: the transformer
writes a tuplet's notes with their plain values and no ratio, so six 16ths under one
beam in a beat become six plain 16ths and the bar runs long. On 26 pieces with exact
MuseScore scores, a quarter of the notes were written that way. Most such groups
print no numeral, and the long bars throw off the meter too. The transformer's
time-signature token holds only the bottom number, so the writer guesses the top
one from the bar lengths: 4/4 written as 3/2. Every page guesses its own, so
18% of bars were written under the wrong meter in the joined score, and 32.6% page by
page.

With `--read-time-signatures` (and `--output-visual-sidecar`), homr reads the time
signature printed on the page, as two stacked numbers, common time or cut time, and
writes it instead of the guess; common and cut time keep their symbols. A read counts only when every staff of the system and
two passes over the page agree, and when the transformer saw a time signature with
the same bottom number there. A program that runs homr page by page can pass the
meter each page ends in (`ProcessingResult.time_signature_at_end`) on to the next
(`XmlGeneratorArguments(time_signature_from_previous_page=...)`), which then starts in
it and restates it unprinted. A time signature the transformer reads where nothing is
printed, as at a key change, is left out.

With `--repair-tuplets` as well, notes the page shows are a tuplet are written as
one. A group under one beam whose plain values add up to three, six or another count
of its smallest value that plain values cannot make is a triplet. The notes under a
bracketed "3", "5", "6" or "7" take its ratio. A bar's repairs apply only under a
printed or carried time signature, never a guessed one. The staves they touch must
fall short or overflow as read, whether groups start after the shortest note or when
the earliest note ends, and add up exactly with them as the bar is written. No staff
may end further off, and at least three bars of the page must add up to the same
length. A beam group is left alone when anything on the page says it may be something
else: a lone short note beside it, uneven spacing, a notehead off its stem, or a
printed value that disagrees with its token. Each time signature and tuplet, and what
became of it, is recorded in the sidecar.

Both options are off unless asked for. On the 26 pieces (123 pages, each page starting
in the meter the page before ended in):

- Bars written under the wrong meter fell from 18.0% to 0.5% in the joined score, and
  from 32.6% to 2.0% page by page. The first page wrote the printed meter in 25 pieces
  instead of 19.
- Notes with the score's exact duration rose from 73.0% to 79.9%, and notes missing
  their tuplet fell from 25.4% to 18.5%.
- Every one of the 2,727 repaired notes the score could judge now has the score's
  value; none was repaired wrongly.
- Measured bar by bar against the scores, 127 staff-bars got better and none got worse.

On 131 pages of 70 video-game pieces with MIDI exports, 98 repaired notes now start
where the MIDI does, and none moved away from it; the MIDI agrees with 95 more either
way and cannot judge the other 17. On the 240-page corpus, 2,896 notes on 33 pages were
repaired; in a sample checked against the page every repair was right but one run that
the page leaves unclear. With the options off, the MusicXML is unchanged on all of
these pages. The rules were found on these same pieces, so on music not yet checked
some repairs may still be wrong.

The page is read while the transformer runs. Together the two options add about 1.5 s
to a page, 8% of its time, measured on 30 pages processed one at a time with a GPU.

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
