Visual sidecar repairs
======================

This page documents every repair marker emitted by the visual sidecar contract
described in :doc:`visual_sidecar` and the repair behavior behind it. It also
covers repairs that alter serialized output without receiving a dedicated
``repair_actions`` entry.

Repairs are post-inference and visual-only. They may change a visual group's
geometry, staff ownership, MusicXML link, ``moment_id``, or ``chord_id``. They do
not change the transformer symbol stream, recognized pitch or duration, or the
MusicXML note set. In particular, a recovered visual group only supplies
pixel-backed geometry for a MusicXML note that already exists.

How to read repair metadata
---------------------------

Four pieces of metadata describe different aspects of a result:

``notes[].alignment_method``
   Explains how a MusicXML note was linked to a visual group. It belongs to the
   link, not to the source geometry.

``visual_groups[].visual_status``
   States whether a group is a strong linked result, a conservatively repaired
   linked result, or an unlinked diagnostic candidate.

``visual_groups[].provenance``
   Describes where the effective notehead candidate came from. It does not state
   whether the link is canonical.

``visual_groups[].repair_actions``
   Is an additive audit trail. Entries can describe mutations, quarantine
   decisions, or the evidence used to prove physical chord identity. Consumers
   must not execute these actions. More than one entry can apply to a group, and
   their array order must not be treated as a public execution order.

The effective drawing geometry is in ``notehead_ellipses``,
``notehead_contours``, ``stem_contours``, and ``bbox``. The
``detected_notehead_contours`` and ``detected_stem_contours`` fields preserve
detector evidence for comparison; ``refined_notehead_contours`` records accepted
source-image notehead fits. All serialized coordinates are in source-image
space.

Repair pipeline
---------------

The builder performs repairs in this order:

#. Recover eligible segmentation candidates that inference excluded, assign
   every candidate to a physical staff from image geometry, refine initial
   noteheads, and reconstruct stems from detected stem fragments.
#. Before matching, quarantine clef artifacts and duplicates, consolidate exact
   duplicates, and merge unambiguous split hollow noteheads.
#. Construct visual moments and align them to transformer token moments in page
   order. Complete structural assignments are locked before individual
   attention matches are considered.
#. Release a chord-member link that lies outside the neighboring moment interval
   while another member remains correctly anchored.
#. Recover still-unlinked recognized chord members, first from one unique unused
   candidate and then, if necessary, from an independently fitted source-image
   contour.
#. Apply the narrowly constrained cross-staff link repair.
#. Jointly refit weak or staff-position-inconsistent noteheads against source
   pixels.
#. Merge remaining split whole-note fragments, resolve physical ``chord_id``
   values, and mark all remaining candidates diagnostic.
#. Serialize effective geometry and normalize only the unreliable notehead
   ellipse angles described below.

Ambiguity stops a repair. It is normal for a MusicXML note to remain unlinked or
for a visual candidate to remain diagnostic when the evidence is not unique.

Visual status values
--------------------

``canonical``
   The link was established by a complete structural moment, or an
   attention-based link was promoted after independent physical chord evidence
   proved it. ``canonical`` does not mean untouched: a canonical group can still
   contain repaired notehead or stem geometry and corresponding actions.

``fallback``
   The group is linked, but at least one conservative recovery or non-structural
   decision was required. Examples include sequence, attention, cross-staff,
   recovered-candidate, transformer-notehead, and pixel-refit results.

``diagnostic``
   The group is not linked to MusicXML and therefore has ``musicxml_id: null``.
   It is retained to explain rejected, merged, duplicate, or otherwise unmatched
   image evidence. Diagnostic candidates do not themselves fail the consistency
   evaluator.

Provenance values
-----------------

``segmentation``
   The group came from the ordinary notehead segmentation used to prepare the
   transformer input.

``recovered_candidate``
   The group came from a raw notehead candidate excluded from normal inference.
   HOMR filters it by staff bounds and staff-scale size, rejects overlap with an
   existing note, and assigns it to the nearest eligible physical staff. It is
   used only by the sidecar and remains ``fallback`` even if structural matching
   later links it.

``merged_fragments``
   The surviving group absorbed another candidate believed to be a fragment of
   the same physical notehead. The corresponding merge action says whether this
   happened before or after matching.

``transformer_recovered``
   No suitable segmentation group existed for a transformer-recognized chord
   member. A linked chord mate supplied the local x-position, staff, staff
   spacing, and diatonic interval; source-image pixels then had to produce an
   acceptable notehead contour at that location. This creates visual geometry,
   not a new MusicXML note.

Alignment methods
-----------------

``structural``
   The token moment and visual moment formed an unambiguous, order-preserving
   match with compatible per-staff membership. Members are paired top-to-bottom
   using token chord order, so recognized pitch is not allowed to exchange two
   visual heads. Complete segmentation groups become ``canonical``; groups with
   ``recovered_candidate`` provenance remain ``fallback``.

``attention``
   A non-structural assignment was made while the symbol had usable transformer
   coordinates. Normally the symbol and candidate were mutually unique nearest
   neighbors, on the expected physical staff, within a local distance bound, and
   consistent with locked page order. The degenerate case of one remaining symbol
   and one remaining same-staff candidate also uses this method. The group is
   ``fallback`` and gets ``attention_aligned`` unless later promoted to
   ``stem_repair``.

``sequence_repair``
   The link was recovered without a complete structural match. This includes a
   unique proven subset of a surplus or incomplete moment, a non-structural
   assignment for a symbol without usable attention coordinates, and transformer
   chord-notehead recovery. Initial sequence assignments get
   ``sequence_repair_aligned``; the chord-recovery paths have their own more
   specific actions instead.

``stem_repair``
   An original ``attention`` link was promoted after final chord resolution found
   independent physical chord evidence. The linked segmentation group becomes
   ``canonical``. Its audit trail can still contain ``attention_aligned`` because
   that records the initial link path.

``cross_staff_repair``
   The transformer emitted an existing note in the wrong grand-staff branch, and
   the unique other-staff visual candidate passed the cross-staff checks below.
   The group remains ``fallback`` and gets ``cross_staff_link_repaired``.

``none``
   No visual group could be linked. The note record has
   ``visual_group_id: null`` and ``match_confidence: 0``.

Geometry repair actions
-----------------------

``refined_stretched_notehead``
   The initial notehead box was abnormally wide, with width greater than twice
   its height. A source-image boundary search found a better ellipse. HOMR
   replaces the effective notehead contour and ellipse while retaining the
   detector contour separately.

``stem_geometry_repaired``
   The effective stem was missing from the note or differed from the note's
   initially attached stem. HOMR selects nearby stem-like fragments, joins
   collinear fragments, and can reconstruct a better upward or downward stem
   only from those pixels. The result is written to ``stem_contours``;
   ``detected_stem_contours`` retains the originally attached fragment.

``pixel_staff_position_repaired``
   A recognized note's visual staff position disagreed with the clef-derived
   expected slot by a small, repairable amount. The expected pitch bounds the
   search, but accepted center, size, ellipse, contour, hollow/filled state, and
   final ``staff_position`` come from source pixels and the physical staff grid.
   A consistently displaced chord with no correct visual anchor is not repaired.

``joint_notehead_refit``
   One or more related weak or displaced noteheads were refitted in one assignment.
   Candidates can come from one musical moment or the same split segmentation
   clump. Fits must have sufficient core and boundary ink, own disjoint core
   pixels, map to every expected staff slot, and improve the existing geometry.
   The update is atomic: if any member fails, none of the target geometries is
   changed. A changed target also gets ``pixel_staff_position_repaired`` and is
   marked ``fallback``.

``transformer_notehead_recovered``
   A transformer-recognized chord member had no reusable visual candidate. HOMR
   predicts a search location from a linked same-staff chord mate and their
   diatonic interval, then creates a visual group only after an independent
   source-image contour fit succeeds. The new group has
   ``provenance: transformer_recovered`` and ``visual_status: fallback``.

``dense_chord_notehead_recovered``
   Qualifies ``transformer_notehead_recovered``. The ordinary fit was blocked by
   neighboring chord-member geometry, so HOMR retried the pixel fit while
   excluding the already linked chord mates from the neighbor mask. The relaxed
   fit still has to find a valid source-image contour.

Candidate consolidation and quarantine actions
-----------------------------------------------

``merged_split_notehead_before_matching``
   Two narrow, touching, hollow candidates on the same physical staff and staff
   position were uniquely consistent with the left and right halves of one head.
   Before matching, the surviving group absorbs both candidates' detected and
   refined contours, receives a fitted combined ellipse and averaged center, and
   changes provenance to ``merged_fragments``.

``merged_split_notehead``
   After matching, an unlinked hollow, stemless fragment touched a linked hollow
   whole-note group at the same staff position. The linked group absorbs the
   closest compatible fragment and changes provenance to ``merged_fragments``.

``merged_into:<visual-group-id>``
   Marks the diagnostic fragment that was absorbed by the named surviving group
   in either split-notehead repair. This is a parameterized action; the text
   after the colon is an actual ``visual_group_id``.

``duplicate_candidates_consolidated``
   Multiple groups were backed by exactly the same centers, sizes, and detected,
   refined, and effective contours. HOMR retains the lexically first visual ID as
   the candidate used for matching and remembers every duplicate staff-position
   interpretation.

``cross_staff_duplicate_consolidated``
   Qualifies ``duplicate_candidates_consolidated`` when the same segmented pixels
   were admitted by overlapping ledger zones on more than one physical staff.
   Structural moment evidence may later select one remembered staff
   interpretation.

``staff_membership_repaired``
   Structural evidence selected a different physical staff for a consolidated
   cross-staff duplicate. HOMR changes ``staff_index`` and, when available, uses
   the duplicate's remembered ``staff_position`` for that staff. This does not
   change ``musicxml_staff_number`` and is distinct from
   ``cross_staff_link_repaired``.

``suspected_duplicate``
   The group was quarantined as duplicate evidence. It is used for rejected
   members of an exact-duplicate cluster and for a much smaller hollow fragment
   near a full notehead when both share the same detected stem and nearly the same
   vertical center. The group becomes ``diagnostic``.

``duplicate_of:<visual-group-id>``
   On an exact duplicate, names the surviving primary candidate. Small
   stem-sharing fragments can have ``suspected_duplicate`` without this more
   specific reference.

``clef_artifact``
   A notehead candidate lay within the clef exclusion radius of a recognized clef
   in transformer coordinates. It is quarantined as ``diagnostic`` before
   matching so it cannot shift later assignments.

``unmatched_candidate``
   No link or more specific quarantine/merge reason was found for the candidate
   by the end of the pipeline. The group becomes ``diagnostic``. HOMR does not add
   a MusicXML note for it because the pixels alone do not establish pitch,
   duration, and voice.

Link recovery actions
---------------------

``attention_aligned``
   Records an initial non-structural link for a symbol with usable attention
   coordinates, as described by ``alignment_method: attention``. This is normally
   a mutually unique local match; it can also be the sole remaining same-staff
   symbol/candidate pair. If physical chord proof later promotes the method to
   ``stem_repair``, this action remains as the history of the original assignment.

``sequence_repair_aligned``
   Records an initial order-preserving fallback assignment described by
   ``alignment_method: sequence_repair``. Surplus visual candidates are not filled
   greedily; a subset must be uniquely established by attention, a common
   physical stem, or relative diatonic shape. A symbol without attention
   coordinates can also be assigned when it and one compatible candidate are the
   only remaining pair.

``transformer_chord_candidate_recovered``
   An otherwise unlinked recognized chord member had exactly one unused existing
   candidate near the staff position predicted from a linked chord mate. HOMR
   reuses that candidate instead of synthesizing new geometry, links it with
   ``alignment_method: sequence_repair``, and marks it ``fallback``.

``cross_staff_link_repaired``
   An otherwise unlinked MusicXML-side symbol was linked to an existing candidate
   on the other physical staff. HOMR requires all of the following:

   * no candidate in the target slot on the declared staff encodes the same
     diatonic pitch;
   * exactly one candidate in the target slot on the other staff encodes the
     symbol's step and octave under that physical staff's active clef;
   * no other symbol can claim that candidate; and
   * the candidate either joins the already linked target moment at the same
     horizontal position or is the single visual moment strictly bracketed by
     the immediately previous and next linked token moments.

   Accidentals do not participate because the sidecar does not associate
   accidental glyphs independently with noteheads. Missing clefs, ambiguous
   pitches or candidates, and ambiguous or unbracketed rhythmic slots remain
   unlinked. The repaired group keeps its observed ``staff_index``; the note
   keeps its original ``musicxml_staff_number`` and uses
   ``alignment_method: cross_staff_repair``.

Physical chord and separation actions
-------------------------------------

``moment_id`` means shared musical onset. ``chord_id`` is assigned only when
image geometry proves that same-staff noteheads form one physical chord. The
following actions explain why a ``chord_id`` was assigned or deliberately not
assigned. They are evidence and grouping decisions, not contour mutations.

``shared_stem_proven``
   Every member shared an owned detected stem component and their notehead
   geometry was compatible with that common stem. HOMR assigned the same
   ``chord_id``.

``structural_chord_proven``
   The members had compact chord geometry, shared a non-null ``moment_id``, and
   every link used ``alignment_method: structural``. HOMR assigned the same
   ``chord_id`` even when no shared stem component was available.

``hollow_column_proven``
   All members were hollow and formed a compact visual column. HOMR assigned the
   same ``chord_id``. This independent evidence can prove a physical whole-note
   chord across differing recognized duration or voice partitions.

``transformer_chord_recovered``
   At least one member had ``provenance: transformer_recovered`` and the completed
   set had compact chord geometry. HOMR assigned one ``chord_id`` and places this
   action on every member of the proved chord, not only the recovered member.

``mixed_duration_stems_separated``
   One visual moment contained more than one recognized base duration. HOMR
   prevents a shared stem alone from combining those duration partitions into a
   physical chord. A compact hollow column can still independently prove a chord
   across the partitions.

``opposed_stems_separated``
   Close simultaneous heads had independent upward and downward stem components
   leaving opposite sides. HOMR leaves their ``chord_id`` values null even if
   segmentation also made one stem appear shared. Their common ``moment_id`` is
   retained because they still sound at the same onset.

Repairs without a dedicated action
----------------------------------

Some normalizations are visible in the final fields but do not add a
``repair_actions`` entry:

* Physical staff lines are robustly fitted from neighboring grid samples and,
  where source-image support is available, refined against bilateral horizontal
  ink. This determines the initial ``staff_index`` and ``staff_position`` and is
  therefore treated as measurement rather than a later repair.
* Visual moments combine close x-columns, plausible shared-stem components, and
  touching conventionally displaced seconds. The result appears in
  ``moment_id``; final chord proof is separately recorded by the chord actions
  above.
* A chord-member attention link outside the interval established by adjacent
  moments is released when another member remains in order. The symbol can then
  be recovered by the candidate or pixel-backed chord paths, or remain unlinked.
* Output ``stem_component_ids`` are emitted only when another compatible group
  of the same base duration shares the component. The serialized component ID is
  duration-qualified so different duration partitions do not become accidental
  chord evidence.
* A fallback ellipse angle, or a suspicious near-zero mask-fit angle on a filled
  head, can be replaced with the median reliable notehead angle for its staff
  group (falling back to the page-wide median). Only the serialized ellipse angle
  is normalized; contours, centers, and ``bbox`` are unchanged.

Viewer and evaluator behavior
-----------------------------

The viewer should draw the effective serialized geometry and follow the
one-to-one ``musicxml_id``/``visual_group_id`` link. It should not infer pitch,
select a different candidate, merge contours, reconstruct stems, or interpret a
repair action as a command.

The evaluator accepts linked groups with ``canonical`` or ``fallback`` status
and reports diagnostic candidates without treating them as invented MusicXML
notes. It gives ``cross_staff_repair`` special geometric handling only when the
note method, group status, and ``cross_staff_link_repaired`` action form the
complete required metadata combination. Other repair actions are an audit trail
and do not relax the one-to-one link or pitch-consistency rules.
