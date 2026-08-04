Visual sidecar v3 and evaluator
================================

HOMR can emit a visual sidecar next to its MusicXML output. Version 3 is the
viewer contract: each pitched MusicXML note is either linked to exactly one
visual notehead group or is explicitly unlinked. Consumers must not infer pitch,
repair links, or synthesize missing noteheads or stems.

The sidecar is written as ``<image>.homr.visual.json`` when visual-sidecar output
is enabled. The evaluator accepts v3 only; there is no v2 compatibility path.

Staff fields
------------

The v3 names distinguish three different concepts:

``staff_group_index``
   Zero-based index of the staff group processed by one HOMR recognition pass.
   A group can contain one or two physical five-line staffs.

``staff_index``
   Zero-based physical five-line staff within the recognition group. It is ``0``
   for the upper or only staff and ``1`` for the lower staff.

``musicxml_staff_number``
   One-based value of the MusicXML note's ``<staff>`` element. It appears on a
   sidecar ``notes`` record rather than a visual group.

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

An unlinked pixel candidate has ``visual_status: diagnostic`` and
``musicxml_id: null``. Diagnostics are reported but do not fail the consistency
evaluation: they can indicate a real note missed by transformer recognition,
but this tool cannot safely manufacture the corresponding MusicXML note.

The v2 ``musicxml_ids``, ``unmatched_musicxml_notes``, and
``unmatched_visual_notes`` fields are not part of v3. Unlinked MusicXML notes are
represented by ``notes[].visual_group_id: null``; diagnostic visual candidates
are represented directly in ``visual_groups``.

HOMR performs general, pixel-supported repairs before serialization. These
include sequence alignment, physical chord grouping, duplicate consolidation,
and recovery of notehead geometry from existing image candidates. Repair
metadata remains available through ``visual_status``, ``provenance``,
``alignment_method``, and ``repair_actions``. Recovery never creates a MusicXML
note and never invents stem geometry without pixel evidence.

Evaluation CLI
--------------

Run inference and evaluate the two generated artifacts with:

.. code-block:: console

   homr-visual-eval score.png
   homr-visual-eval score.png --report score.visual-eval.json

The command exits with:

* ``0`` when the MusicXML and sidecar agree;
* ``1`` when evaluation finds a note or contract divergence; or
* ``2`` when inference, artifact loading, or v3 validation cannot run.

The machine-readable report lists each divergence and the IDs involved. It
distinguishes missing sidecar records, MusicXML notes without usable visual
noteheads, extra sidecar records, artifact pitch mismatches, visual staff-position
mismatches, and malformed links.

Pitch validation has two independent layers. First, ``notes[].pitch`` must equal
the MusicXML pitch, including its accidental. Second, MusicXML step and octave,
the active clef (including ``clef-octave-change``), and
``musicxml_staff_number`` must imply the visual group's ``staff_position``.
The second check detects a pitch assigned to the wrong printed line or space.

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

Curated image-corpus evaluation and the autonomous test-and-fix loop are a
separate follow-up. Once consumers require v3, viewer-side pitch overrides,
link repairs, and synthetic visual fallbacks can be removed so the viewer treats
the sidecar as authoritative.
