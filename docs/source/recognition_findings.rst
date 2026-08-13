Recognition-quality findings
============================

This document records changes that have demonstrated an improvement in homr's
recognition quality. Each finding should describe the original behavior, the
change, the evidence, and the configuration under which the result was
observed. New findings should be added below rather than replacing earlier
results.

Finding 1: merge SegNet class scores before selecting a class
-------------------------------------------------------------

Status
~~~~~~

Accepted. Keep this correction enabled. The recommended configuration is the
fixed merger with a 320-pixel SegNet step, which means no intentional overlap
between tiles (apart from the shifted tiles needed to cover the final page row
or column).

Problem
~~~~~~~

SegNet operates on a fixed ``320 x 320`` window and returns six per-class
score maps for each window. The legacy reconstruction selected a class inside
each tile first, producing an integer class-label image, and then averaged
those labels in page coordinates. That operation is not a valid way to merge
class predictions. It can create a class that no tile predicted: for example,
the average of class IDs ``1`` and ``3`` is ``2``.

The problem is especially relevant at tile boundaries. A symbol or staff line
can receive different predictions from neighboring windows. Once the labels
have been collapsed to integers, the merger has lost the relative evidence
for every class and cannot make a meaningful decision between those
predictions.

Correction
~~~~~~~~~~

The implementation in
``homr/segmentation/inference_segnet.py::merge_patches`` now performs the
following operations:

#. Retain each tile's complete ``(class, height, width)`` score tensor.
#. Accumulate every class score into a floating-point page-sized array at the
   tile's page coordinates.
#. Accumulate a page-sized coverage array at the same time.
#. Divide the accumulated class scores by coverage, giving each contributing
   tile equal weight.
#. Apply ``argmax`` once, after page-level score merging, to obtain the final
   class map.

The model window remains fixed at 320 pixels. This finding changes only the
representation and timing of the merge decision; it does not change SegNet or
the transformer weights. The output class mapping remains the existing one:

* class 1: stems and rests
* class 2: noteheads
* class 3: clefs and key signatures
* class 4: staff
* class 5: symbols

Class 0 remains the background class.

The reconstruction grid is shared by inference and merging. It includes the
shifted final row and column when a page dimension is not an exact multiple of
the window or step, and it handles pages smaller than the model window.

Focused validation
~~~~~~~~~~~~~~~~~~

``tests/test_inference_segnet.py`` covers:

* one tile reconstructing unchanged;
* non-overlapping tiles reconstructing unchanged;
* identical overlapping scores;
* conflicting overlaps selecting the highest averaged class score;
* preventing an unrelated class from being invented by averaging class IDs;
* shifted final rows and columns;
* pages smaller than 320 pixels; and
* output dimensions matching the input page exactly.

The HOMR test suite passed after the correction: 193 tests passed, with the
existing warnings and six subtests also passing. The focused SegNet tests and
the relevant inference checks were included in that run.

Corpus evaluation
~~~~~~~~~~~~~~~~~

The comparison used the recorded pitch-reference corpus with the same source,
models, rasterization, and provider settings for both runs:

* 31 PDF pages, rasterized with pypdfium2 at 300 DPI, RGB, white background;
* ``--gpu auto``; the recorded run used CPU execution because CUDA and CoreML
  were unavailable;
* cache disabled and visual sidecars enabled;
* identical HOMR source hash:
  ``71e79aad48444571c4ebba9a15dd7e09a1a78e19ed173aa9a92acc536581b828``;
* identical SegNet model hash:
  ``6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d``; and
* identical transformer encoder and decoder model hashes, recorded in each
  run's provenance.

The run identities are:

* legacy merger:
  ``omr-evals/runs/pitch-reference/pitch-baseline-legacy-merge``;
* corrected merger:
  ``omr-evals/runs/pitch-reference/pitch-fixed-merger-step-320``.

The aggregate results were:

.. list-table:: Legacy versus corrected merger
   :header-rows: 1
   :widths: 22 16 16 16 16 16 16

   * - Run
     - Exact pages
     - Diverged pages
     - Error pages
     - Total errors
     - Error rate
     - Wrong / missing / extra
   * - Legacy merger
     - 27
     - 4
     - 0
     - 8
     - 0.000956
     - 4 / 3 / 1
   * - Corrected merger, step 320
     - 28
     - 3
     - 0
     - 5
     - 0.000597
     - 3 / 2 / 0

At the page level, one page improved, 30 were unchanged, and no page
regressed. The corrected run therefore reduced total errors by three, reduced
the note error rate by 0.000359, produced one additional exact page, removed
one wrong pitch, removed one missing note, and removed one extra note. Neither
run had a trill error.

The result is a recognition-quality improvement, not merely a segmentation
unit-test improvement. The corrected merger was the first tested change in
this work that improved the corpus recognition score.

Super Mario case study
~~~~~~~~~~~~~~~~~~~~~~

On
``slightly-flawed/super-mario-bros-ground-theme``, page 1, the legacy output
had three errors:

* one missing E4;
* one extra E4; and
* one wrong pitch in measure 17, where the reference was E5 and the output was
  E-flat 5.

The corrected merger produced an exact pitch result for all 208 reference
notes on that page. In particular, the measure-17 note changed from E-flat 5
to E5.

Why a SegNet fix can change a transformer pitch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pitch is emitted by the TrOMR transformer, but the transformer does not
read the original page directly. The relevant pipeline is:

.. code-block:: text

   SegNet class maps
       -> notehead, stem, staff, and symbol detection
       -> staff geometry and dewarping
       -> transformer staff crop
       -> transformer encoder and decoder
       -> pitch token and MusicXML

The merger correction changes the binary structural masks. Those masks affect
which noteheads and stems are detected, how staffs are fitted, and how each
staff image is cropped and dewarped before transformer inference. A small
change in that image can move the transformer across a decision boundary even
though its weights and decoder are unchanged.

The saved Super Mario artifacts support this explanation:

* the rasterized page PNG is byte-identical between the legacy and corrected
  runs;
* the target visual note remains at the same reported center and has the same
  note identity;
* its upstream stem component changes from ``staff-4-stem-72`` in the legacy
  artifact to ``staff-4-stem-65`` in the corrected artifact; and
* the corrected MusicXML emits E5 for that note instead of E-flat 5.

The visual sidecar records downstream geometry and component identities, not
the raw SegNet score tensors. Therefore the exact per-pixel boundary decision
that caused this pitch change cannot be reconstructed from the saved corpus
runs alone. An instrumented future run should persist the merged class map,
the per-class score margin at changed pixels, the detected staff geometry, and
the transformer staff input image when a page's MusicXML changes.

Overlap follow-up and scope of the finding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After isolating the merger correction, overlap was tested separately using the
same mean-class-score merger. These experiments do not invalidate the fix;
they test a different variable. The comparison was against the corrected
step-320 run, not against the legacy merger:

.. list-table:: Fixed merger overlap sweep
   :header-rows: 1
   :widths: 18 12 14 20 18 18

   * - Step
     - Overlap
     - Exact / diverged / error
     - Total errors
     - Error rate
     - Paired page result
   * - 320
     - 0%
     - 28 / 3 / 0
     - 5
     - 0.000597
     - Baseline
   * - 240
     - 25%
     - 24 / 6 / 1
     - 12
     - 0.001463
     - 0 improved, 25 unchanged, 5 regressed, 1 failure
   * - 160
     - 50%
     - 22 / 9 / 0
     - 23
     - 0.002748
     - 0 improved, 24 unchanged, 7 regressed

The smaller-step runs show that uniform averaging of the current raw score
maps is not a safe overlap policy for this model. They are not evidence that
the class-score merger correction should be disabled. The accepted finding is
the class-score merge itself, with the controlled step-320 configuration.
Center-weighted blending and probability-normalized merging remain separate
future experiments and must not be conflated with this finding.

Operational conclusion
~~~~~~~~~~~~~~~~~~~~~~

Keep ``MERGE_STRATEGY = "mean-class-scores"`` enabled and use
``segmentation_step_size = 320``. Treat the legacy integer-label merger as a
historical baseline only. Future recognition improvements should be recorded
as additional findings with a reproducible run identity, page-level deltas,
and a clear statement of which upstream or downstream stage changed.
