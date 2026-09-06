# R8-AD5 Result — Causal Motion Intervention

**Frozen classification:** **K0 — neither preregistered causal contrast supported**

Run: `34013934993`

Authoritative raw/frozen result: [`../results/r8_ad5/aggregate/FINAL_RESULT.md`](../results/r8_ad5/aggregate/FINAL_RESULT.md)

## Question

AD3/AD4 established that recent native path history improves simple linear task accessibility, that normalized displacement direction preserves about 94% of the one-step accessibility gain, and that a vector-valued second difference adds further accessibility. AD5 asked whether those readable motion features were also preferentially causal under matched transition interventions.

The experiment explicitly spliced the next state of a frozen mature trajectory. The pre-splice state was identical across arms. Direction rotations and magnitude-only changes were matched to the same Euclidean displacement from the native next state. A second test compared turn-orientation changes against in-plane scalar-turn changes while matching step norm and splice size.

## Primary result

Both preregistered causal gates failed.

### Direction versus magnitude

Frozen contrast:

`K_DIR = DROP_DIR - DROP_MAG`

where positive values would mean rotating the native transition direction is more damaging than changing only its length at the same splice norm.

Observed:

- `K_DIR` mean **-0.016792** (-1.68 pp)
- bootstrap 95% CI **[-0.022087, -0.011519]**
- positive in **1/12** lineages
- cross-entropy contrast mean **-0.044629**, positive in **1/12**

Mean terminal accuracy drops:

- direction rotation: **+1.92 pp**
- magnitude-only: **+3.60 pp**
- generic norm-matched random: **+2.01 pp**

Thus matched magnitude changes were substantially more disruptive than direction rotations. Generic random perturbations behaved much more like the direction-rotation arm than the magnitude-only arm.

The reversal was not confined to one perturbation scale:

- `rho=0.05`: mean `K_DIR = -0.005793`
- `rho=0.10`: mean `K_DIR = -0.016251`
- `rho=0.20`: mean `K_DIR = -0.028331`

It was also negative at every preregistered splice time `t = 2,4,6,8,10`.

### Turn orientation versus scalar turn

Frozen contrast:

`K_CURV = DROP_AZ - DROP_POL`

where positive values would mean changing the azimuth/orientation of the turn is more damaging than changing the scalar turn angle in the existing turn plane at matched step norm and splice displacement.

Observed:

- `K_CURV` mean **-0.030533** (-3.05 pp)
- bootstrap 95% CI **[-0.040005, -0.020050]**
- positive in **1/12** lineages
- cross-entropy contrast mean **-0.237109**, positive in **1/12**

Mean terminal accuracy drops:

- turn-orientation / azimuth intervention: **+4.80 pp**
- polar / scalar-turn intervention: **+7.85 pp**

The contrast was negative for both frozen azimuth angles (15° and 30°) and at every frozen splice time.

Maximum frozen geometry mismatch across all families was `3.744e-06`, below the preregistered `1e-5` validity tolerance. All 12 lineages were valid.

## What AD5 establishes

AD5 does **not** support the preregistered claim that native displacement direction or turn orientation is preferentially causal under these matched transition splices.

Instead, it establishes a useful negative boundary:

> **A feature can be highly informative to a simple diagnostic reader without being the direction of greatest causal vulnerability for the trained computation.**

In AD4, normalized displacement direction preserved about 94% of the one-step linear-accessibility gain while scalar displacement magnitude preserved only about 5%. In AD5, however, perturbing transition magnitude was substantially more damaging than rotating the transition direction at the same Euclidean splice size.

That is a direct readability–causality dissociation in the tested system.

## Exploratory interpretation — not a promoted result

The reversal suggests a new mechanism worth testing, but AD5 itself does not establish it.

A plausible interpretation is **tangent/phase sensitivity with transverse robustness**:

- a direction rotation is predominantly a transverse displacement away from the native transition;
- a magnitude-only change moves the state forward/backward along the native transition direction;
- the recurrent flow may correct transverse deviations while retaining a phase/timing offset along the native trajectory;
- because the terminal reader is evaluated after a fixed number of recurrent steps, an along-flow phase shift could be more damaging than a transverse deviation that is subsequently corrected.

This interpretation is compatible with the observation that generic random perturbations caused about the same terminal damage as direction rotations, whereas trajectory-tangent magnitude perturbations were more damaging.

It must be tested directly before being promoted.

## Next justified test

The clean next experiment is a **tangent-versus-transverse recovery / phase persistence** diagnostic:

1. apply norm-matched tangent and transverse perturbations at the same native state;
2. track perturbation separation for every subsequent recurrent step;
3. decompose the deviation into tangent and transverse components relative to the native flow;
4. test whether transverse error contracts/rejoins while tangent error persists as a phase-like displacement;
5. relate persistence to terminal functional damage;
6. keep the AD5 K0 outcome frozen regardless of what the follow-up finds.

If tangent deviations persist while matched transverse deviations contract, that would explain the AD5 reversal without rescuing the failed direction-causality hypothesis.

## Claim boundary

AD5 does not establish information beyond the complete Markov state-plus-map, state-independent motion information, essential chronology, a universal trajectory code, or practical superiority. The phase/tangent account above is exploratory until directly tested.
