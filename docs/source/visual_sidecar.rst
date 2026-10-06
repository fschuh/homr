Visual sidecar and evaluator
============================

HOMR can emit a visual sidecar next to its MusicXML output.

This documentation covers visual sidecar version 3 only. It is the viewer
contract: each pitched MusicXML note is either linked to exactly one visual
notehead group or is explicitly unlinked. Consumers must not infer pitch, repair
links, or synthesize missing noteheads or stems.

The sidecar is written as ``<image>.homr.visual.json`` when visual-sidecar output
is enabled. The evaluator enforces this contract.

Staff fields
------------

The field names distinguish four different concepts:

``staff_group_index``
   Zero-based index of the staff group processed by one HOMR recognition pass.
   A group can contain one or two physical five-line staffs.

``staff_index``
   Zero-based physical five-line staff within the recognition group. It is ``0``
   for the upper or only staff and ``1`` for the lower staff.

``musicxml_staff_number``
   One-based value of the MusicXML note's ``<staff>`` element. It appears on a
   sidecar ``notes`` record rather than a visual group. It normally identifies
   the same staff as ``visual_groups[].staff_index + 1``. The sole exception is
   an explicitly marked cross-staff repair described below.

``staff_position``
   Diatonic line/space position local to the physical staff. The bottom line is
   1, the bottom space is 2, and the top line is 9. Ledger positions continue
   below 1 or above 9. HOMR calculates this value from the notehead center and
   detected staff-line geometry, independently of transformer pitch.

Link contract
-------------

Each entry in ``notes`` uses a unique ``musicxml_id`` and has a singular
``visual_group_id`` or ``null``. Each linked entry in ``visual_groups`` has the
inverse singular ``musicxml_id``, a ``moment_id``, notehead geometry, and a
``visual_status`` of ``canonical`` or ``fallback``. The two directions must
agree, and neither identifier can participate in more than one link.

``moment_id`` represents a shared musical onset and must agree for members of a
MusicXML chord. ``chord_id`` is optional and narrower: it groups noteheads only
when image geometry proves one physical visual chord. Simultaneous noteheads
with independent or opposed stems therefore share a ``moment_id`` while keeping
``chord_id: null``.

An unlinked pixel candidate has ``visual_status: diagnostic`` and
``musicxml_id: null``. Diagnostics are reported but do not fail the consistency
evaluation: they can indicate a real note missed by transformer recognition,
but this tool cannot safely manufacture the corresponding MusicXML note.

Unlinked MusicXML notes are represented by ``notes[].visual_group_id: null``;
diagnostic visual candidates are represented directly in ``visual_groups``.

Repair metadata
---------------

HOMR applies pixel-supported visual repairs before serialization. The result is
already authoritative: consumers must not replay ``repair_actions`` or infer a
different link from them. The metadata explains how the effective geometry,
link, staff membership, moment, or chord identity was obtained.

See :doc:`visual_sidecar_repairs` for the ordered repair pipeline, the complete
``visual_status``, ``provenance``, ``alignment_method``, and ``repair_actions``
vocabulary, and the evidence required by each repair.

Evaluation CLI
--------------

Run inference and evaluate the two generated artifacts with:

.. code-block:: console

   homr-visual-eval score.png
   homr-visual-eval score.png --report score.visual-eval.json

Evaluate existing artifacts without loading models or rerunning inference:

.. code-block:: console

   homr-visual-eval --musicxml score.musicxml
   homr-visual-eval --sidecar score.homr.visual.json
   homr-visual-eval --musicxml score.musicxml --sidecar alternate.visual.json

With only ``--musicxml``, the evaluator infers
``<stem>.homr.visual.json`` in the same directory. With only ``--sidecar``, it
infers ``<stem>.musicxml`` by removing the standard ``.homr.visual.json``
suffix. The inferred absolute path is printed before the files are loaded. If
the inferred file does not exist, evaluation exits with status ``2``. Supplying
both paths overrides inference; artifact options cannot be combined with the
positional image mode.

The command exits with:

* ``0`` when the MusicXML and sidecar agree;
* ``1`` when evaluation finds a note or contract divergence; or
* ``2`` when inference, artifact loading, or sidecar validation cannot run.

The machine-readable report lists each divergence and the IDs involved. It
distinguishes missing sidecar records, MusicXML notes without usable visual
noteheads, extra sidecar records, artifact pitch mismatches, visual staff-position
mismatches, and malformed links.

Pitch validation has two independent layers. First, ``notes[].pitch`` must equal
the MusicXML pitch, including its accidental. Second, MusicXML step and octave,
the active clef (including ``clef-octave-change``), and the relevant physical
staff must imply the visual group's ``staff_position``. Normally that physical
staff is specified by ``musicxml_staff_number``. For an explicitly marked
cross-staff repair, the evaluator instead uses the clef belonging to the visual
group's ``staff_index``. The second check detects a pitch assigned to the wrong
printed line or space.

Accidental glyphs are not yet represented and associated independently in the
sidecar. Consequently, the geometric check validates diatonic step and octave,
while accidental correctness is limited to agreement between the sidecar pitch
string and MusicXML. Unsupported or missing clef evidence makes the evaluation
fail as unevaluable instead of guessing.

Scope
-----

This evaluator measures MusicXML/sidecar consistency and viewer usability; it is
not a ground-truth optical-recognition evaluator. In particular, a note omitted
by transformer recognition can be absent from both MusicXML and linked sidecar
notes. Any corresponding pixel candidate remains diagnostic and does not become
an ``added_visual_note`` failure.
Optional annotation geometry
----------------------------

Schema v3 may additionally contain ``annotation_geometry`` with ``version: 1``
and a nonempty ``staffs`` array. Each physical staff has ``staff_id``,
``staff_group_index``, ``staff_index``, and ``system_index``. The first two
numeric staff fields match visual-group physical membership; ``system_index``
groups staffs by printed system independently of voice traversal order.

``lines`` contains five top-to-bottom polylines sharing an increasing-x sample
grid. ``spacing`` contains [x, vertical staff-space] samples on the same grid.
``extent`` is [left, top, right, bottom]. All coordinates use the original source
raster, through the same crop/resize transform as note geometry. Consumers must
not extrapolate outside the extent. A display raster resized independently in x
and y must scale line coordinates, extents and spacing (spacing uses y scale).

The optional capability is ``physical-staff-curves-v1``. Absence does not
invalidate v3 note links or normal viewing; it makes annotation layout
unavailable. Cache readiness must inspect this capability, not just sidecar v3
or the HOMR package version. Targeted page regeneration can upgrade old pages;
blanket cache invalidation is not required. Invalid advertised geometry is a
contract error, including missing staff identities referenced by visual groups.

Rest verification
-----------------

Schema v3 may additionally contain ``rest_verification`` with ``version: 1`` and a
``rests`` array: one record for every rest in the MusicXML, linked by ``rest_id`` to
the ``id`` of its ``<note>``. Rest IDs are distinct from note IDs. A record says
whether a printed rest backs that MusicXML rest.

``status`` is one of:

``supported``
   Free-standing ink of a rest's shape was found where the transformer placed the
   rest. ``center`` is that ink's centre.

``unsupported``
   No such ink was found, or the rest exists only because a note was read without a
   pitch. ``center`` is where the transformer placed it, on the staff's middle line.
   A consumer can show it as a rest that is not printed.

``unverified``
   The rest could not be checked; ``reason`` says why.

``reason`` is ``rest_shaped_ink``, ``no_rest_shaped_ink``,
``ink_claimed_by_another_rest`` (the only nearby rest ink was the better match for
another rest), ``note_without_pitch``, ``multi_measure_rest``,
``no_attention_coordinates``, ``no_staff_lines``, ``no_staff_geometry``,
``no_segmentation``, or ``not_verified`` (the rest never reached the verifier).

``duration`` is the transformer token, for example ``rest_8``, or ``note_16`` for a
note read without pitch. ``staff_lines`` holds the five line heights of the rest's
physical staff at its ``center`` x, top to bottom, and ``unit_size`` one staff space;
both are empty or null when the rest could not be placed. ``position_estimated`` is
true when the transformer gave the symbol no position and its x was taken midway
between its neighbours in reading order. ``part``, ``measure``,
``musicxml_staff_number``, ``voice``, ``staff_group_index`` and ``staff_index`` have
their meanings above. Coordinates use the original source raster, like note geometry;
a display raster resized independently in x and y scales ``unit_size`` and
``staff_lines`` by the y scale.

Segnet has no rest class, so the verifier looks for ink that no segmentation class
explains. The block is diagnostic only: beyond the rest IDs that link its records, it
never changes the MusicXML or any note link, and its absence does not affect them.

Note value verification
-----------------------

Schema v3 may additionally contain ``note_value_verification`` with ``version: 1`` and
a ``notes`` array: one record for every MusicXML note linked to a visual group that is
not ``diagnostic``, keyed by ``musicxml_id``. A record compares the note's recognized
value with the value printed on the page.

``printed`` is the value read from the page, one of ``whole``, ``half``, ``quarter``,
``eighth``, ``16th`` and ``32nd``, or null when the reader is not sure. ``dotted`` says
whether an augmentation dot is printed, or is null when that is not sure. Tuplets are
compared by their plain value: a triplet eighth is an eighth.

``status`` is one of:

``agrees``
   The printed value is the recognized one, and so is the dot when it was read.

``disagrees``
   The printed value, or a dot that was read, differs from the recognized one. A
   consumer can mark the note.

``unknown``
   The printed value was not read; ``reason`` says why.

``reason`` names the evidence for a reading: ``0_bands`` to ``3_bands`` (flags or beams
counted at a filled notehead's stem), ``hollow_head_with_stem`` or
``hollow_head_without_stem``. Otherwise it says why there is none:
``not_a_plain_note`` (a grace note), ``notehead_unclear`` (neither clearly hollow nor
clearly filled), ``no_stem``, ``stem_unclear`` (no single stem leaves the notehead, as
where two voices share it), ``bands_unclear``, ``too_many_bands``, ``two_voices``
(hollow and filled heads at one moment), ``no_staff_lines``, ``staff_too_small`` or
``no_segmentation``.

The notes of a chord share a stem, its flags or beams and its dots, so they are read
together and get the same ``printed`` and ``dotted``. The reader reports a value only
when every check it relies on agrees, and abstains otherwise. The block is diagnostic
only: it never changes the MusicXML or any note link, and its absence does not affect
them.

Timing repairs
--------------

Two voices that sound one pitch at the same moment often share a printed notehead:
one voice's stem rises from it and the other's falls from it, each with its own flags
or beams, as when an eighth note sits on the first note of a 16th run. The transformer
gives such a head one token, usually with the longer value, so the MusicXML writer
starts the next notes when that value ends and the rest of the bar runs late.

When the visual sidecar is written, the writer reads both stems of every notehead that
stands alone on its staff at its moment. Where one stem carries a shorter value than
the token, it starts the next notes when that shorter value ends, as if the lost note
were still sounding. No note is added, and the token keeps its value and its notehead.
A bar's repairs are applied together, and only when the bar does not add up to its
time signature as recognized and adds up exactly with them. The time signature's beat
count is itself measured from the typical bar, so the repairs may change it only to a
length that a bar needing no repair already has. A bar with an applied repair always
places its groups when the earliest sounding note ends, the rule the repair is defined
by, even where starting them after the shortest note would leave its staves closer to
the barline. ``--no-timing-repairs`` turns the repair off; the MusicXML is then written
exactly as without it.

Schema v3 then contains ``timing_repairs`` with ``version: 1``, ``enabled`` (whether
the repair was on) and a ``shared_noteheads`` array, with one record, in writing order,
for every token whose notehead was read with two stems:

``musicxml_id`` and ``visual_group_id``
   The note and its notehead. ``musicxml_id`` is null when the token is not linked.

``part`` and ``measure``
   Where the note is written.

``recognized``
   The transformer's token, e.g. ``note_8``.

``printed_up`` and ``printed_down``
   The values read on the stem rising from the head's right edge and on the stem
   falling from its left edge, or null where that stem's flags or beams were not read.
   ``dotted`` is as in note value verification.

``next_notes_after``
   The shorter printed value the next notes start after, or null where no repair was
   proposed.

``status`` and ``reason``
   ``applied`` with ``bar_adds_up``; ``declined`` with ``bar_already_adds_up``,
   ``bar_still_does_not_add_up``, ``no_bar_length`` (no time signature, and the typical
   bar is not a whole number of eighths), ``token_matches_neither`` (both stems were
   read and the token carries neither value), ``dotted``, ``values_unclear`` or
   ``not_a_plain_note`` (dotted, tuplet or grace tokens); or ``not_needed`` with
   ``token_is_shortest``.

The block is absent when the writer never ran with the sidecar, and holds an empty
array with ``enabled: false`` when the repair was off.
