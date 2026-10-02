# Current focus: bounded flow-SiD feasibility study

Decision: October 2, 2026. The user postponed the expensive competitive comparison
and chose to learn from the existing toy experiment. Program 2 remains deferred.

## Assignment and bounds

Continue the already-authorized frozen teacher/student campaign. Keep the existing
teacher, eight SiD/SiDA arms at 1/2/3/4 NFE, 4,000-update student horizons, recipes,
selection/confirmation rules and finite watchdogs. Report 1/2 NFE before continuing
3/4 as previously authorized. This decision does not order a stop, add runs or
extend any horizon. No further teacher training, FID-target chasing or recipe search.

The question is: **Does this pixel-space flow adaptation show useful learning,
and does the adversarial package improve the quality/diversity tradeoff at this
fixed exploratory budget?** Competitive FID remains contextual, not an acceptance
gate. Failure to beat Sangwoo or the TA does not by itself invalidate the pilot.

## Extract findings from the measurements already planned

| Question | Evidence | Limit on the conclusion |
|---|---|---|
| Does the adaptation learn useful generation? | Within-arm 1k/2k/4k FID trajectory, fixed-noise class-reference galleries, numerical/gradient health | Finite gradients alone are insufficient; no convergence claim from 4k updates |
| Does adversarial training help here? | A−S FID at matched checkpoints and exposure, selected and final outcomes, confirmation draws, diversity and nearest-neighbor evidence | Full adversarial-package effect under this teacher, not an isolated loss coefficient |
| Do more calls help at this budget? | Actual NFE, final outputs, per-position training exposure and measured cost | Equal cycles do not give equal per-position exposure or equal compute |
| Is any benefit meaningful rather than collapse or copying? | Recognizable class structure, independent noise variation, duplicates/saturation and nearest optimization examples | Pixel variance and a few attractive examples are not semantic coverage |

Compare matched-update trajectories as well as best selected checkpoints; a single
lucky checkpoint is weaker evidence than a persistent direction that survives the
already-declared second sampling seed. Additional sampling seeds are not independent
training replications. Use existing plots and galleries, with broad class review
for any positive claim; do not build another monitoring platform.

When judging improvement over initialization, use a same-NFE initial baseline if
one was actually measured. Otherwise mark it unavailable. The teacher's 32-call
FID is context, not an equivalent one-call initialization baseline. Do not compare
student quality to an unrelated untrained plumbing checkpoint as a scientific control.

## Decisions and interpretation

A promising result is a coherent learning trend and useful class-conditioned
variation, or a paired adversarial benefit supported by both quality and diversity
evidence. A mixed result (for example, lower FID but less structural variation)
is worth reporting. A weak or flat result can remain inconclusive because both
teacher and student exposure are limited. Numerical failure or broad collapse is
a finding about this implementation/recipe, not proof the method is impossible.

These interpretation rules were written after S1's first evaluation became
available; they are not a retrospectively pre-registered success threshold.
Keep all arms and negative outcomes in the report. Do not retrofit criteria to
make a favorite method win or launch additional experiments to obtain a success.

## Compact handoff for possible follow-up by Sangwoo

At the existing 1/2-NFE milestone and final report, produce a short local summary:

1. One sentence stating the question and main observation.
2. A table of teacher FID/exposure and each completed student's NFE, G/fake/real
   exposure, selected/final/confirmation FID, diversity findings and training cost.
3. Two or three representative matched class panels, including failures.
4. Exact source/config/partition/checkpoint identities and the material differences
   from Sangwoo's EDM setup; distinguish observed evidence from speculation.
5. If warranted, one concrete longer-training test he could consider and what
   outcome would support or weaken our interpretation. No scaling recommendation
   based only on stable losses or one favorable picture.

This is preparation for a possible discussion, not authorization to send Slack
messages or edit Notion. The paired pilot cannot prove that teacher distillation
protects against GAN collapse without an adversarial-only control, and comparison
with the earlier AFM collapse does not supply a matched control.
