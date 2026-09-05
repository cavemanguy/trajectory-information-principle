# ND-R1 Post-run Audit

**Authoritative run:** GitHub Actions run `33932823763`  
**Pinned implementation commit:** `ed80a9fd60daeaf012b5384b2d1fdb0841b8246b`  
**Fresh primary seeds:** 13, 29, 53  
**Frozen primary classification:** **Outcome A — training reproduction failure**

This audit is post-primary. It does not change the preregistered ND-R1 outcome.

## 1. Frozen primary result

ND-R1 preregistered a competence gate requiring epoch-100 mean h12 validation accuracy >= 0.50 in every fresh seed.

Fresh results:

| Seed | h12 val @100 | C0 | G0 | G100 | Delta G | Delta G 95% CI | selectivity seed criterion |
|---:|---:|---:|---:|---:|---:|---|---|
| 13 | 0.19735 | -6.7681 | 0.0181 | 0.8280 | +0.8099 | [0.7908, 0.8181] | PASS |
| 29 | 0.19920 | -7.1788 | 0.0245 | 0.8112 | +0.7867 | [0.7590, 0.7966] | PASS |
| 53 | 0.18545 | -7.3989 | 0.0164 | 0.6179 | +0.6014 | [0.5808, 0.6071] | PASS |

Because all three seeds failed the frozen h12 >= 0.50 competence gate, the authoritative classification is Outcome A even though all three passed the separately preregistered selective-preservation seed criterion.

The outcome is not retroactively changed.

## 2. Post-run competence-gate audit

After the ND-R1 outcome was known, the preserved original Observer-R2 core artifacts were inspected to determine whether the 0.50 h12 threshold was actually calibrated to the source lineage.

Historical Observer-R2 core artifacts:

- seed 7 artifact `9821298916`
- seed 19 artifact `9821996148`
- seed 43 artifact `9823283251`

At each original core's selected epoch 100 checkpoint:

| Historical seed | mean h0 val | mean h12 val | original combined validation metric |
|---:|---:|---:|---:|
| 7 | 0.6452 | 0.2142 | 0.4297 |
| 19 | 0.6122 | 0.2583 | 0.43525 |
| 43 | 0.67555 | 0.2496 | 0.462575 |

The historical lineage itself therefore did **not** approach 0.50 mean h12 accuracy. ND-R1's 0.50 h12 competence gate was miscalibrated to the recovered source architecture.

For comparison, the fresh ND-R1 epoch-100 combined `(h0+h12)/2` validation metrics were approximately:

- seed 13: 0.4196
- seed 29: 0.4105
- seed 53: 0.4123

These are somewhat below but close to the historical combined range 0.4297–0.4626.

### Scientific consequence

Outcome A remains the formal ND-R1 result because the gate was frozen before execution.

However, it should **not** be paraphrased as "the fresh models failed to learn anything comparable to the Observer lineage." The more precise conclusion is:

> ND-R1 failed its preregistered competence gate, but post-run provenance audit shows that the gate demanded a level of h12 competence the historical source lineage itself did not possess.

A new independent confirmatory experiment is required with a lineage-calibrated competence gate; ND-R1 cannot be rescued by changing its threshold after seeing results.

## 3. Strong post-primary candidate: training-amplified relation specialization

Despite the formal Outcome A, the native natural-pair analysis produced a striking reproducible secondary pattern.

At initialization, relation-specific terminal survival was almost indistinguishable across all eight relations:

- seed 13: G0 = 0.0181
- seed 29: G0 = 0.0245
- seed 53: G0 = 0.0164

By epoch 100:

- seed 13: G100 = 0.8280
- seed 29: G100 = 0.8112
- seed 53: G100 = 0.6179

The increase was large and bootstrap-positive in all three seeds.

The amplified relations differed by seed:

- seed 13: relation 2 survival 2.529; relation 7 survival 1.846; most others ~0.28–0.41
- seed 29: relation 3 survival 2.511; relation 0 survival 2.040; most others ~0.31–0.43
- seed 53: relation 1 survival 2.799; most others ~0.37–0.59

Because the synthetic relation channels are statistically symmetric at the task level while the identity of the strongly preserved relation changes across seed, this is consistent with **training-amplified symmetry breaking / spontaneous specialization** rather than a fixed semantic ordering of relations.

This is a **post-primary candidate phenomenon**, not an ND-R1 confirmed primary claim.

## 4. Early establishment of final preservation ranking

The preregistered secondary correlation between relation survival after two recurrent transitions and final relation survival was:

- seed 13: Spearman 0.9762
- seed 29: 0.9286
- seed 53: 0.9048
- mean: 0.9365

Thus most of the final relation-survival ordering was already visible by the second recurrent transition in all three fresh seeds.

This strongly echoes the historical R10 observation that much of relation-survival ranking was established very early in recurrence.

It remains secondary in ND-R1.

## 5. R2-style transient observations

### Indirect trajectories replicate; reversal dominance does not universally replicate

At epoch 100, mean endpoint/path-length efficiency was low in all fresh seeds:

- seed 13: 0.1618
- seed 29: 0.1228
- seed 53: 0.0823

The trajectories therefore remained highly indirect relative to straight endpoint displacement.

However, reversal fraction differed dramatically:

- seed 13: 0.0395
- seed 29: 0.9451
- seed 53: 0.9965

Therefore **reversal-dominated motion is not a seed-general property** of this fresh reproduction, even though indirect trajectories are.

### Earliest direction remains much more informative than final direction

Epoch-100 deterministic ridge accessibility:

| Seed | first direction | integrated direction | final direction |
|---:|---:|---:|---:|
| 13 | 0.4181 | 0.3484 | 0.1584 |
| 29 | 0.3974 | 0.2975 | 0.1597 |
| 53 | 0.4397 | 0.3971 | 0.1831 |

The first direction is strongest in all three fresh seeds. This is consistent with the earlier R3 caution that much apparent directional-history accessibility can be dominated by the earliest transient rather than chronology or accumulated ordered history.

### Native state accessibility collapses quickly

Epoch-100 ridge accessibility from h0 was approximately 0.477–0.491, while h12 was approximately 0.168–0.185. Much of the generic linear accessibility disappears during recurrence even though selected relation distinctions can be geometrically amplified.

This reinforces the need to keep the following separate:

- generic information accessibility;
- relation-specific geometric survival;
- native reader performance;
- trajectory geometry.

## 6. Practical relevance

The strongest practical clue from ND-R1 is not perturbation-based retrieval.

It is that training can appear to turn an almost uniformly contractive recurrent map into a **selective dynamical filter** in which a small subset of otherwise symmetric task channels is preferentially preserved or amplified.

If independently confirmed under a correctly calibrated training gate, practical branches could test:

- dynamically selective filtering/compression;
- learned sparse channel preservation;
- low-cost recurrent preprocessing for small downstream readers;
- controlled symmetry breaking as a routing/allocation mechanism;
- early-step computation, since final preservation ranking is largely established by transition 2.

These are engineering hypotheses, not claims established by ND-R1.

## 7. Required next scientific step

Do not change ND-R1.

Create a new fresh-seed experiment with:

1. the same no-perturbation natural-pair survival endpoint;
2. the same recovered Observer-core architecture;
3. a competence gate calibrated **before execution** from the preserved historical source lineage rather than from ND-R1 results;
4. completely new seeds;
5. the symmetry-breaking/specialization pattern declared in advance as a secondary confirmatory endpoint.

This preserves the negative gate result while testing whether the striking selective-preservation phenomenon survives a properly calibrated independent confirmation.
