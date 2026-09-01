# Research History: From Trajectories to Active Latent Interrogation

This is a record of the experimental path, including negative and ambiguous results. It is not a claim that Active Latent Interrogation (ALI) was inevitable, nor that every earlier experiment tested the same mechanism.

The approximate progression was:

> trajectory information → internal observer → active perturbation → latent geometry → learned query-conditioned addressing → causal direction interventions

## How to read this record

### Plain English

Some experiments produced measurements, some produced suggestive patterns, and some were only prototypes. Those are different levels of evidence. Failed ideas remain here because they helped eliminate simpler explanations and narrow the question.

### Technical evidence labels

- **Measured observation:** a value or controlled comparison was recorded.
- **Interpretation:** an explanation consistent with the measurement but not uniquely established by it.
- **Negative result:** the proposed mechanism failed to improve the target metric or performed worse than a control.
- **Mixed result:** effects differed across tasks or manipulations.
- **Design/prototype only:** code or protocol exists, but no verified outcome is available in the present record.
- **Abandoned hypothesis:** retained historically but not supported as a current claim.

Some values below were recovered from earlier experiment-summary artifacts that are not yet checked into this repository. They are transcribed research history, not presently reproducible results. Decimal accuracies are left in their original form to avoid implying extra precision.

## 1. Trajectory information

### Plain English

The starting idea was that two states can finish at the same endpoint while taking different routes, and that the route may retain information lost at the endpoint. Early work used hand-built attractor-like dynamics and trajectory features. That intuition was useful, but the implementation did not justify the broad claims originally attached to it.

### Technical description

The early question was whether a path functional

$$s=\rho(z_0,z_1,\ldots,z_T)$$

could predict an initial condition or label when terminal state $z_T$ could not. The archived Python prototype embedded the input directly into its initial coordinates and stored trajectory signatures for lookup-style recovery. This demonstrates that deterministic paths can differ; it does not establish compression, general recovery, convergence guarantees, or a universal trajectory-information principle.

A later C benchmark used contracting and rotating dynamics in which states were designed to approach the same zero endpoint after 40 steps. It defined final-state, compressed-path, and full-history comparisons. The recovered source description contains no verified printed outcome values.

**Evidence status:** the Python demonstration is historical code, not support for current claims. The same-endpoint C benchmark is **design/prototype only** until outputs and an evaluation script are recovered.

**Abandoned hypothesis:** that convergence trajectories generally preserve enough information to reconstruct initial conditions, or that stable-attractor dynamics provide inherent robustness.

## 2. Internal observer and UnitDirection experiments

### Plain English

The next step was to place a small observer inside the recurrent computation. Instead of assuming the whole trajectory was useful, the observer accumulated selected information about motion while the main memory evolved. UnitDirection-style features emphasized *which way* the state moved rather than only how far it moved.

### Technical description

For recurrent states $m_{t+1}=f(m_t)$, observer variants received features such as normalized motion

$$u_t=\frac{m_{t+1}-m_t}{\lVert m_{t+1}-m_t\rVert_2+\varepsilon}$$

and updated an observer state $o_{t+1}=g(o_t,u_t,m_t)$. Readouts compared geometric/motion observer information with final-memory information and ablated individual pathways.

Recovered strong-ablation summaries reported:

| GeometricObserver condition | Accuracy | Change from native | Predictions changed |
|---|---:|---:|---:|
| Native | 0.5566 | — | — |
| Observer zeroed every step | 0.5181 | -0.0385 | 32.15% |
| No observer feedback | 0.5452 | -0.0114 | 14.48% |
| No final memory in readout | 0.5223 | -0.0342 | 27.88% |

MotionObserver ablations reportedly produced smaller effects.

**Measured observation:** the geometric observer and final memory both affected predictions. Zeroing the observer caused a larger decrease than removing observer feedback in this summary.

**Interpretation:** the observer carried task-relevant information not fully duplicated by the final state. The smaller no-feedback effect does not show feedback was useless, but it weakened the idea that recurrent feedback was the primary mechanism.

**Limitation:** an observer ablation can establish dependence on an information path, not that the path represents trajectory geometry in a unique or general sense.

## 3. Path order, working trajectories, and feedback

### Plain English

Experiments then asked whether the *order* of internal observations mattered. Reversing or shuffling the path hurt, but shifting it halfway actually helped. That was important: the model cared about ordering, yet the native ordering was not automatically best.

Other attempts tried to turn trajectories into an active workspace or feed observer information back into memory. Those ideas did not become reliable mechanisms.

### Technical description

Recovered ordered-path summaries reported:

| Ordered-path condition | Accuracy |
|---|---:|
| Native replay | 0.5814 |
| Half-shift | 0.6108 |
| Shuffled | 0.5547 |
| Reversed | 0.5279 |

Reversal reportedly changed 53.98% of predictions.

**Measured observation:** ordering manipulations affected the output; shuffle and reversal reduced accuracy relative to native replay, while half-shift improved it.

**Interpretation:** the observer/readout was order-sensitive. These values do not support a stronger claim that the learned or physical trajectory order was uniquely correct, because a non-native shift performed best.

**Negative/mixed result:** working-trajectory variants and observer-feedback variants did not produce a consistent scaling or retrieval advantage. The recovered observer ablation above also shows only a modest native-versus-no-feedback difference (0.0114) in that setting.

**Abandoned hypothesis:** that maintaining an explicit working trajectory, or recursively feeding observer output back into the dynamics, was by itself the missing scalable memory mechanism.

## 4. Active perturbation and frozen-memory response

### Plain English

The research then changed the role of the observer. Instead of passively watching whatever motion happened, the system deliberately nudged a frozen memory and measured its response. Freezing the memory made the question cleaner: could information be retrieved from the reaction without rewriting the stored representation?

### Technical description

Given frozen memory $m$, a probe direction $v$, and magnitude $\alpha$, experiments measured a response such as

$$r=F(m+\alpha v)-F(m).$$

Response-only controls restricted the decoder to $(r,q)$ rather than handing it $m$ directly. Frozen-memory controls prevented the probe training stage from adapting the memory encoder or response dynamics.

**Measured observation:** response-only models achieved above-control task performance in later experiments, so the perturbation response contained decodable task information.

**Interpretation:** active perturbation can expose information in a frozen representation. It does not establish that the information was inaccessible to every direct readout, nor that the effect requires long trajectories rather than local directional sensitivity.

**Narrowing step:** this moved the project away from universal claims about naturally occurring trajectories and toward a controlled intervention with an explicit information boundary.

## 5. Latent geometry and direction scans

### Plain English

Direction scans showed that the memory did not respond equally in every direction. Some nudges produced more useful responses than others. Perturbations also tended to shrink over recurrent steps. That made direction selection look more important than simply observing a long trajectory.

### Technical description

Geometry scans varied unit direction $v$ while controlling perturbation norm and inspected response magnitude or retrieval performance. Directional anisotropy means that for some $v_i,v_j$,

$$\lVert J_F(m)v_i\rVert \ne \lVert J_F(m)v_j\rVert$$

or that the resulting responses differ in task information. Recurrent contraction was observed when perturbation separation

$$d_t=\lVert z_t^+-z_t\rVert$$

decreased across steps.

**Measured observation:** direction mattered, and perturbation differences contracted over recurrent evolution in the explored systems.

**Interpretation:** the useful signal may be a direction-dependent local or short-horizon response. Contraction argues against assuming that longer recurrent rollout preserves or amplifies probe information.

**Limitation:** anisotropy is common in nonlinear learned representations. Observing it does not by itself establish a special latent geometry or a retrieval mechanism.

## 6. Learned versus fixed, orthogonal, and diverse probes

### Plain English

A natural guess was that probes should spread out and point in very different directions. The experiment succeeded at making them diverse—but retrieval got worse. Diversity itself was not the answer. Letting the system learn useful directions without forcing orthogonality worked better.

### Technical description

Forced-diversity probes reduced mean probe cosine to approximately 0.21, compared with approximately 0.97 for unconstrained probes. The recovered summary reported:

| Probe policy | Native accuracy | Accuracy with response noise $\sigma=0.2$ |
|---|---:|---:|
| Forced diverse | 0.5939 | 0.5811 |
| Unconstrained learned | 0.6228 | 0.6044 |

Gaussian-matched, shuffled, and zero controls were reported near chance.

**Measured observation:** the diversity objective changed geometry as intended but reduced task accuracy, both without and with response noise.

**Negative result:** forced orthogonal/diverse directions were not a successful retrieval mechanism in this experiment.

**Interpretation:** useful directions may be clustered because tasks share sensitive subspaces, or high cosine may arise for another reason. The experiment rejects “more angular diversity is automatically better”; it does not prove why learned probes cluster.

## 7. Learned query-conditioned addressing

### Plain English

The next hypothesis was that the question itself should choose the nudge. Results were mixed across tasks: query conditioning helped on a three-relation task but lost badly on a compositional task. This prevented a general claim and pushed the work toward tighter controls.

### Technical description

A query-conditioned policy/readout was compared with a final-state head. Recovered summaries reported:

| Task | Query-conditioned | FinalStateHead | Outcome |
|---|---:|---:|---|
| ThreeRelation | 0.6027 | 0.5191 | Query-conditioned higher |
| Compositional | 0.6585 | 0.7576 | Final-state head higher |

**Measured observation:** the relative result changed by task.

**Mixed/negative result:** query-conditioned addressing was not a general win and did not solve compositional scaling.

**Interpretation:** query-dependent directions could be useful under some memory/task structures, but direct state information remained stronger in at least one harder setting.

## 8. Current frozen-memory experiment and causal direction interventions

### Plain English

The current experiment asks a narrower question: in a frozen compressed memory, does selecting the native probe direction matter when the decoder sees only the response and query? Damaging direction selection caused larger drops than the small difference between probing and direct controls.

### Technical description

On the current 8-relation synthetic task, reported accuracies are:

| Main condition | Accuracy |
|---|---:|
| Native query-directed probing | **64.67%** |
| Wider direct control | 63.60% |
| Direct frozen-state readout | 62.40% |
| Query-blind probing | 62.15% |

Direction interventions reported:

| Direction condition | Accuracy |
|---|---:|
| Native query-directed | **64.67%** |
| Wrong query for probe selection only | 56.94% |
| Shuffled directions | 51.29% |
| Mean direction | 50.29% |

**Measured observation:** performance in this trained system depends strongly on native direction assignment.

**Interpretation:** the learned probe direction is example- and query-relevant rather than an arbitrary perturbation.

**Unresolved confound:** the current policy is $v=P(h,m,q)$ and may inspect memory while constructing $v$. The intervention therefore does not isolate query-only addressing.

**Next control:** train $v=P(q)$ while keeping the readout restricted to response and query. This is the next genuine narrowing step, not a guaranteed confirmation.

## 9. What survived the narrowing process

### Plain English

The surviving claim is smaller than the original trajectory idea: a learned, query-conditioned direction can matter when probing a frozen latent memory. We still do not know whether the query alone can select that direction or whether looking at memory during probe construction is essential.

### Technical conclusion

The history supports studying controlled directional response under strict information boundaries. It does not currently support:

- universal trajectory information preservation;
- guaranteed convergence or inherent robustness;
- forced probe diversity as a useful principle;
- working trajectories or observer feedback as scalable mechanisms;
- a general advantage over direct readout;
- compositional scaling;
- replacement of attention;
- established novelty.

The current falsifiable question is whether query-only addressing $v=P(q)$ preserves useful retrieval performance when $R$ receives only $(r,q)$.
