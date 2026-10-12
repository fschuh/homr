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
were still in its group. No note is added, and the token keeps its value and its
notehead. A bar's repairs are applied together, and only when the bar does not add up
to its time signature as recognized and adds up exactly with them, each placed as the
writer places the bar: after the shortest note, or by the earliest end where a trusted
meter lets the bar prove that (see ``homr.bar_placement``). The time signature's beat
count is itself measured from the typical bar, so the repairs may change it only to a
length that a bar needing no repair already has. ``--no-timing-repairs`` turns the
repair off; the MusicXML is then written exactly as without it.

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

Printed time signatures
-----------------------

The transformer's time-signature token carries only the beat type
(``timeSignature/4``), and the writer infers the beat count from the typical bar
length: one for the page, or with ``--guess-meter-per-section``
(``XmlGeneratorArguments(guess_meter_per_section=True)``) one for the bars each token
governs. Missed tuplets make those bars too long, so the inferred meter goes wrong with
them, as 4/4 written as 3/2.

With ``--read-time-signatures`` (``XmlGeneratorArguments(read_time_signatures=True)``,
which needs the visual sidecar), homr reads the time signatures printed on the page
while the transformer runs: two stacked numbers inside the staff, common time or cut
time, found in the page's ink with the staff lines taken out and read with OCR. A read
is used only when every read of it agrees (each staff of its system, and two passes
that take out the staff lines at their exact thickness and a pixel thicker), when its
numbers are a meter (1 to 32 beats of 1, 2, 4, 8, 16 or 32), and when the transformer
put a time-signature token with the same beat type within six staff spaces of it. The
read meter then replaces the inferred beat count for the token's section, and common and
cut time are written as their symbols (``<time symbol="common">`` and ``symbol="cut"``).

A caller writing a score page by page passes the meter the previous page ended in as
``time_signature_from_previous_page``; ``ProcessingResult.time_signature_at_end``
gives it for the next page. The bars before the page's first time signature are then
measured against it, and it is written into the first measure with
``print-object="no"``, unless the page starts with a time signature of its own.

The transformer also reads time signatures where none is printed, as at a key change.
Under a trusted meter (one read on the page or carried to it), a token with nothing
shaped like a time signature printed near it is not written, and its bars keep the
trusted meter. A token where something is printed that no read could make out ends
the trusted meter: its section's beat count is inferred, as without the reader, and
the page then ends in no trusted meter.

Schema v3 then contains ``time_signatures`` with ``version: 1``:

``carried_in`` and ``carried_out``
   The meter passed in from the previous page and the trusted meter the page ends in,
   each ``[beats, beat_type]`` or null.

``tokens``
   One record per time-signature token: ``point`` (where the transformer saw it, in
   source pixels, or null), ``beat_type``, ``status`` and ``read``. ``status`` is
   ``read`` (a trusted read lands on it), ``beat_type_differs`` (one lands on it with
   another beat type), ``unread_glyph`` (something is printed there that no read made
   out) or ``no_read`` (nothing shaped like a time signature is printed there).
   ``read`` is the read that landed on it, or null.

``unmatched_reads``
   Trusted reads no token landed on. Each read, here and in ``tokens``, has
   ``system_index``, ``x``, ``span`` (the system's top and bottom line), ``beats``,
   ``beat_type`` and ``symbol`` (``common``, ``cut`` or null for numbers).

The block is absent when the time signatures were not read.

Tuplet repairs
--------------

The transformer often writes a tuplet's notes with their plain values and no ratio:
six 16ths under one beam in a beat come out as six plain 16ths, and the bar runs long.
With ``--repair-tuplets`` (``repair_tuplets=True``, which needs the time signatures
read), the writer repairs such groups where the page shows them:

* a beam joins the notes and their plain values add up to three, six or another
  count of their smallest value that no plain values make. The group is then a
  triplet; a beam alone implies no other ratio;
* a bracketed numeral ("3", "5", "6" or "7") spans the notes, whose values, as printed
  where the note-value reader is sure of them, add up to that many of their smallest
  value. A numeral without its bracket is not read.

A beam group is not repaired when a lone note an eighth or shorter stands next to it,
when its stems are unevenly spaced (the widest gap more than 1.6 times the narrowest,
or 2.2 times when its values differ), when a notehead of one of its chords lies more
than a staff space off the chord's stem, when a printed value the reader is sure of
disagrees with its token, or, in a compound meter, when it is a whole number of dotted
beats, or it fills the whole bar.

A bar's repairs apply together, and only under a trusted meter (read on the page or
carried to it; never an inferred one), when every staff they touch falls short or
overflows as recognized and adds up exactly with them, no staff ends further from the
bar length, and at least three bars of the page under the same meter add up as the
writer places them, each with its own repairs. A repaired note gets the tuplet's
token, e.g. ``note_12`` for a triplet eighth, so the MusicXML writes its
``time-modification``, and tuplet start and stop marks where the writer can bracket
the group.

Schema v3 then contains ``tuplet_repairs`` with ``version: 1``:

``numerals``
   The page's bracketed numerals: ``system_index``, ``x``, ``y``, ``number``,
   ``above`` and ``bracket`` (the x of its two hooks), in source pixels.

``groups``
   The beam groups and numerals the repair weighed: ``source`` (``beam`` or
   ``numeral``), ``bar`` (counted from the page's first bar), ``staff``, ``notes`` and
   ``outcome``. ``candidate`` groups carry their ``ratio``. The others say why not:
   ``lone_short_neighbour``, ``uneven_spacing``, ``head_off_stem``,
   ``value_disagrees``, ``rest_or_single``, ``not_plain`` or ``units_differ``. Beam
   groups whose values add up plainly, already a tuplet, or explained by the meter are
   not listed.

``notes``
   One record per note of a candidate group: ``musicxml_id`` and ``visual_group_id``,
   ``bar``, ``source``, ``recognized`` and ``repaired`` (the tokens), ``bar_length``,
   ``applied``, and ``reason`` when not applied: ``no_trusted_meter``,
   ``adds_up_as_recognized``, ``not_exact_with_repairs``, ``a_staff_gets_worse`` or
   ``too_few_supporting_bars``.

The block is absent when the repair did not run.
