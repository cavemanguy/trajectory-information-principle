# Observer Program: R2–R11

This document records the mechanistic observer/recurrent-dynamics program that followed ALI-N8-R1. It is separate from the ALI-N8-R1 claim and should not be read as evidence that ALI replaces attention or that trajectory dynamics create new information.

## Scope

Across R2–R11 the recurring question became narrower: how does a learned recurrent latent system change the *accessibility and geometric form* of task information that is already present in its state?

The central claim boundary throughout is:

> deterministic recurrence can reformat, contract, expose, or hide information under a given observer class without creating new task information independent of the complete earlier state.

## R2 — online geometric observer

A recurrent observer restricted to local trajectory geometry accumulated task-relevant information beyond a matched instantaneous geometric snapshot and reset control. The effect replicated across seeds, but dependence on the native temporal ordering failed to replicate. This supported history-dependent accessibility under the restricted observer, not a unique chronology mechanism.

## R3 — observer-independent directional history

Directional trajectory information was recoverable by readouts other than the original observer. A simple permutation-invariant sum of directions performed strongly, while the first direction alone already carried substantial signal. Exact chronology was therefore not necessary for most of the observed decodability.

## R4 — origin of the first-direction signal

The first displacement was largely an approximately linear, anisotropic re-expression of the already-informative initial state. Untrained and randomized transition operators could preserve or even increase first-direction decodability. Training was not uniquely required to make the first direction informative.

## R5 — later directional reformatting

Later directions retained relation-specific residual structure after restricted prediction from the initial state. Realized recurrent history made later motion easier to predict under restricted model classes than the initial state alone. Evolving Jacobian geometry explained part of this reformatting, and a small intermediate intervention causally altered later directional accessibility while leaving overall task content largely intact.

## R6 — computational function

The strongest supported interpretation was **readout preparation**. Across recurrence, the trained final reader became substantially better while independent linear task accessibility became worse. Interference reduction, target/nuisance separation, selective amplification, useful task-selective compression, and a separate routing mechanism were not established.

A small relation-specific intervention derived from R5 residual geometry causally shifted the matching downstream reader margin with little change in overall task content.

## R7 — reader versus recurrence adaptation

New readers could readily adapt to frozen recurrence. Recurrence also showed reader-conditioned plasticity: with encoder and reader controlled, retrained recurrence could move terminal representations toward an externally altered reader geometry. However, arbitrary fixed-reader systems did not cross the locked competence threshold, so general recurrence adaptation toward arbitrary readers was not established.

## R8 — emergence during training

The R6 readout-preparation regime was reproduced from scratch with dense checkpoints. Training was multiphase: architectural contraction existed before learning; early readout preparation emerged during successful joint training; recurrent-core plasticity was necessary for the normal early preparation gain; later transition-specific directional structure matured after initial reader compatibility.

## R9 — reversibility and selective empirical collapse

Forward prediction through recurrence was easy, while held-out inversion became much harder. Strong inverse models did not restore the original initial-state accessibility geometry. The recurrent map remained locally full numerical rank along sampled trajectories but was extremely ill-conditioned and strongly volume-contracting.

Different task distinctions survived by very different amounts. Training changed an almost indiscriminately destructive initialized recurrence into one that selectively preserved some distinctions. The result supports selective empirical recoverability loss, not fundamental information destruction.

## R10 — geometry of selective preservation

R10 established a bidirectional causal mechanism for distinction survival. Training produced encoder–recurrence coadaptation: the recurrent core learned a strongly anisotropic preservation/contraction field, while the encoder learned relation-dependent placement within that field.

Most relation-survival ranking was established in the first two recurrent transitions. Exact evolving Jacobian propagation predicted relation survival closely.

Holding perturbation norm and first-order relation-semantic projection approximately fixed:

- rotating a normally preserved distinction toward locally contractive geometry reduced downstream survival;
- rotating a poorly preserved distinction toward locally favorable geometry rescued survival.

This established **orientation relative to local dynamical geometry → distinction survival**. It did not establish that increasing survival improves the final reader.

## R11 — does survival mediate downstream use?

R11 tested the missing causal arrow.

The R10 survival manipulation reproduced across all three seeds. The initial full-pair R11 manipulation failed the locked semantic-preservation envelope on validation and was rejected before test evaluation. A preregistered fallback used the R10 micro-intervention family with a locked strength ladder. The largest validation-valid strength was epsilon = 0.01.

At that strength, immediate matching-relation linear prediction changes stayed around 1%, full-task MLP prediction changes stayed below 1%, and immediate reader-margin changes were small.

### Rescue

Low-survival favorable-orientation interventions increased geometric survival relative to semantic-matched random controls by approximately:

- seed 7: +1.708x
- seed 19: +1.071x
- seed 43: +1.038x

But the primary matching-minus-off-relation reader effect was negative in all three seeds, with paired bootstrap intervals below zero.

### Suppression

High-survival contractive-orientation interventions reduced geometric survival by approximately:

- seed 7: -0.893x
- seed 19: -0.906x
- seed 43: -0.865x

Yet the primary reader effect was slightly positive in all three seeds.

Controlled-pair reader-space separation showed the same opposite-sign pattern. Near-boundary examples did not rescue the hypothesis. Independent terminal probes and newly trained small readers did not reproducibly benefit from the added geometric survival. A validation-derived late native-template orientation correction also failed to make rescue useful.

The preregistered classification is **Outcome G — shared-cause association**.

The narrow conclusion is:

> Geometric survival of a task-associated perturbation can be increased or decreased dramatically while immediate task representation remains closely matched, without producing the corresponding improvement or impairment in the frozen downstream reader. The natural association between relation survival and reader usefulness is therefore not explained by Euclidean survival magnitude alone. Training appears to co-organize survival geometry and reader-compatible terminal format.

## Current mechanistic picture

The strongest combined picture through R11 is:

> distributed task structure in the encoded state  
> + intrinsically contractive recurrent dynamics  
> + learned encoder/recurrent coadaptation  
> → orientation-dependent selective survival  
> → recurrent reformatting toward a reader-compatible terminal representation

But **selective survival magnitude itself is not the demonstrated mediator of reader use**.

A recurring result across the program is that these concepts must remain separate:

- information content;
- observer-relative accessibility;
- geometric survival;
- terminal reader alignment;
- causal importance.

## Evidence status

R11 compact preregistration, protocol amendment, primary evidence table, and result summary are stored under `experiments/observer_r11/` and `results/reproducible/observer_r11/`.

The complete generated R8–R11 working bundles and raw arrays are not all checked into this repository. Therefore this repository should not claim complete end-to-end reproduction of the entire R2–R11 sequence from main alone. The compact records are intended to preserve the scientific conclusions, controls, provenance boundaries, and negative results without rewriting the historical path.
