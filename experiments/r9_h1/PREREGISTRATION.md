# R9-H1 Preregistration — Hybrid Interaction Anomaly Screen

**Frozen before implementation outcomes are inspected.**

## Research intent

R9-H1 is an anomaly-hunting experiment. Its purpose is not to optimize speed, parameter efficiency, or benchmark accuracy. Task performance is treated primarily as a validity check: an internal dynamical phenomenon is more interpretable when the arm is demonstrably engaged with the task.

The central question is:

> What reproducible, task-relevant dynamical organization appears when recurrent and attention-based systems are arranged in different serial, reverse-serial, parallel, supervisory, interrogative, and learning-coupled relationships?

This experiment preserves the project boundary that trajectory information is not assumed to be an independent substance beyond the complete state and transition map. It also does not assume that any particular geometry, phase, attractor, tangent direction, oscillation, or module specialization must emerge.

## Benchmark

Use the frozen R9-T1 episodic regime-key temporal task and generator semantics without outcome-driven task redesign. Each episode defines random per-episode regime keys, switches regimes, inserts distractors, and later reactivates earlier regimes from regime ID alone.

All arms receive equivalent raw task information. Any architecture-specific side channel must be explicitly described below and must not provide direct target labels.

## Fresh family seeds

Eight fresh family seeds, disjoint from R9-T1:

`[2311, 2333, 2357, 2381, 2411, 2437, 2467, 2491]`

Training, validation, test, probe, ablation, and intervention streams use deterministic disjoint seed namespaces.

## Architecture / interaction arms

### A. Serial R-T-R

`RNN -> Transformer -> RNN`

- front recurrent module contextualizes the continuously driven stream;
- causal Transformer performs long-range mixing / associative retrieval;
- back recurrent module receives Transformer output and evolves a persistent state.

### B. Reverse serial T-R-T

`Transformer -> RNN -> Transformer`

- causal front Transformer contextualizes the stream globally;
- recurrent middle module evolves a persistent state from that representation;
- causal back Transformer reads the recurrently transformed sequence.

### C. Parallel R || T

Raw event embeddings drive a recurrent stream and a causal Transformer stream in parallel. A small learned fusion head receives both streams. Neither branch receives the other branch during the main forward path.

Primary anomaly question: do the two branches spontaneously become complementary, redundant, or asymmetrically specialized?

### D. Transformer-above-RNN perturbation teacher

A causal Transformer observes the same legal history plus a bounded projection of the RNN state and emits a low-dimensional perturbation vector `p_t` that is applied to the RNN hidden state.

The perturbation channel is bottlenecked and norm-limited. It is not supervised to encode the target label.

Two evaluations are required:

1. **teacher-on**: perturbations remain active;
2. **teacher-off**: Transformer perturbations are removed after training.

The scientific question is whether training under structured perturbation produces persistent changes in the RNN's autonomous dynamics or merely external control.

### E. Transformer interrogator

A Transformer receives a query/context representation and emits a bounded probe direction `p_t`. The probe perturbs the RNN state. A response feature is formed from the difference between perturbed and unperturbed future RNN evolution under matched future inputs.

The readout may use the response feature but the Transformer may not directly write the answer into the classifier input.

Primary question: can a learned controller discover probe directions whose dynamical responses expose task-relevant distinctions beyond matched random, shuffled-query, and wrong-query controls?

### F. RNN-conditioned attention controller

The RNN does not replace the Transformer output. Instead, a bottleneck projection of recurrent state modulates Transformer attention logits through a bounded additive attention bias / gate.

Primary question: can continuously evolving recurrent state organize where attention looks, and is any benefit/history dependence destroyed by temporal shuffling or state replacement?

### G. RNN-conditioned low-rank Transformer modulation

The recurrent state generates bounded low-rank adapter coefficients for selected Transformer linear maps:

`W_eff = W_base + A diag(c_t) B^T`

where `A` and `B` are learned global low-rank factors and `c_t` is a small bounded vector generated from recurrent state.

This is a fast-weight / hypernetwork-like control arm. It is included as a mechanism probe, not a novelty claim.

Primary question: does recurrent trajectory history organize transient Transformer computation in a reproducible way?

### H. Learning-coupled RNN -> Transformer update gate

During training only, an RNN summarizes legal temporal context and emits bounded coefficients that scale a small preregistered set of auxiliary update gates / loss weights for Transformer submodules. The controller cannot inspect target labels except through the same training loss signal available to the base learner.

At evaluation the controller is either frozen-on or removed according to the frozen evaluation protocol.

Primary question: can temporal state organize plasticity rather than merely inference?

### I. Reciprocal / mutual teaching arm

RNN and Transformer are trained on the same task with separate primary predictions and small cross-teaching losses on bottlenecked latent summaries. Each module receives the other's stop-gradient teaching target but not unrestricted hidden-state access.

Primary question: do the modules converge to redundant codes or differentiate into complementary temporal / associative roles?

## Required controls

Where applicable, every controller / perturbation arm includes:

- matched random control;
- temporally shuffled controller output;
- wrong-query or wrong-context control;
- frozen-random controller control;
- norm-matched perturbation control;
- bottleneck capacity matched control;
- teacher/controller removal at evaluation when meaningful;
- bypass ablation of the controlled module;
- target-leak audit.

No result may be interpreted as dynamical organization if the same effect is reproduced by a control carrying equivalent capacity without the relevant temporal relationship.

## Minimal task validity floor

R9-H1 is not a performance competition. Accuracy is not the primary endpoint.

For an arm to receive **strong mechanistic interpretation**, it must satisfy both:

- mean held-out DATA accuracy >= 0.20;
- mean held-out RETURN_EARLY accuracy >= 0.15.

These thresholds are deliberately modest and are not claims of task mastery. Arms below the floor are preserved and reported but their latent anomalies are labeled exploratory / low-competence and cannot support a strong task-relevant mechanism claim.

There is no ranking gate requiring one architecture to beat another.

## Primary anomaly measurements

All measurements are preregistered as descriptive / mechanistic screens. No single metric defines trajectory information.

### 1. Local history accessibility

For each eligible internal state stream compare closed-form probes using:

- state only;
- state + one-step displacement;
- state + multi-step displacement summary;
- state + shuffled displacement;
- displacement only.

Report task-relevant hidden-variable decoding and answer decoding separately.

### 2. Module-unique accessibility

For multi-module arms fit matched-capacity probes on:

- RNN state alone;
- Transformer state alone;
- concatenated states;
- concatenated states with one branch shuffled across examples;
- cross-stream difference / relation features.

A complementary-accessibility anomaly is present only if the joint representation gives a reproducible gain over both individual branches and over shuffled-pair controls.

### 3. Cross-module transformation

Measure whether episode key / regime information becomes more or less accessible at each boundary:

`input -> module 1 -> module 2 -> module 3`.

This is a reformatting/accessibility claim, not information creation.

### 4. Reactivation transient

Around `ACTIVATE(r)` events, measure per-step recovery of the hidden active key and task output. Compare native order to shuffled-order, reversed-local-history, first-transient-only, final-transient-only, and integrated/order-independent summaries where applicable.

Chronology is claimed only if order-sensitive controls fail to explain the effect.

### 5. Path dependence / hysteresis

Construct matched task contexts reached through different legal histories. Compare internal states, future predictions under matched continuations, and probe accessibility.

A geometric difference alone is not sufficient; report functional consequence separately.

### 6. Perturbation response organization

For perturbation-capable arms compare learned probes/controllers with matched random, shuffled, and wrong-query controls. Measure response norm, response direction similarity, downstream functional effect, and cross-state generalization.

Do not force or privilege tangent/transverse geometry.

### 7. Serial-order asymmetry

Compare R-T-R and T-R-T at homologous boundaries. The target is not which is faster or more accurate but whether module order produces reproducibly different internal organization after controlling for task engagement and approximate capacity.

### 8. Parallel specialization

For R || T measure:

- individual branch decodability;
- unique contribution under branch ablation;
- redundancy / complementarity via matched probes;
- trajectory-history gain per branch;
- fusion dependence on each branch;
- cross-seed consistency versus seed-specific specialization.

### 9. Teacher-induced persistence

For perturbation-teacher and learning-coupled arms compare teacher/controller-on versus teacher/controller-off evaluation and a never-taught baseline.

A persistent-teaching candidate requires a reproducible change in the student's autonomous behavior or internal accessibility after teacher removal. Immediate performance with teacher present is insufficient.

### 10. Learning-trajectory coupling

For the update-gated arm record frozen coarse training statistics at fixed checkpoints: loss, gradient norms for preregistered modules, controller coefficients, and parameter-update norms.

Test whether controller state predicts or organizes subsequent update structure beyond time-step / loss-magnitude controls. This is exploratory unless independently confirmed.

## Anti-cheating / answer-channel audits

For perturbation, modulation, and teaching arms:

- estimate the standalone target decodability of the controller signal;
- train a matched probe from controller signal to answer on held-out data;
- compare to legal context-only baselines;
- cap controller dimensionality and norm;
- prohibit direct target-label input;
- report whether replacing the student with a shallow decoder over the controller signal recovers task performance.

If the controller signal itself trivially carries the answer, classify the affected mechanism as **answer-channel confounded** and do not interpret it as learned dynamical interrogation or teaching.

## Frozen screen classifications

The experiment produces a vector of classifications rather than one winner.

For each architecture / interaction arm:

- `V+`: passes minimal task validity floor;
- `V-`: below validity floor.

For each anomaly family:

- `A+`: preregistered effect is positive across >= 6/8 seeds and bootstrap 95% CI excludes zero in the predicted comparison;
- `A0`: no clear reproducible effect;
- `A-`: reproducible effect in the opposite direction;
- `AX`: invalid/confounded.

For phenomena without a directional prediction, use a preregistered nonzero contrast and require consistency across >= 6/8 seeds plus CI excluding zero.

No architecture is declared globally superior from this screen.

## Outcome handling

1. Smoke test validates shapes, determinism, causal masking, artifact schema, and target-leak tests only.
2. Smoke output may not be used to modify scientific thresholds.
3. Eight fresh family seeds run after smoke passes.
4. Do not inspect individual family outcomes while other seeds are running.
5. Frozen aggregation runs only after all eight families finish.
6. Read the frozen aggregate classifier before opening seed-level results.
7. Preserve all negative, opposite-sign, invalid, and confounded outcomes.
8. No outcome-driven training extension, architecture rescue, seed replacement, or metric substitution.
9. Experimental results remain off `main` unless explicitly authorized.

## Claim boundaries

Even strong R9-H1 anomalies would not establish:

- an independent hidden phase variable;
- information beyond complete state-plus-map;
- universal trajectory coding;
- architectural novelty of recurrence/attention hybrids, fast weights, hypernetworks, or learned perturbation systems;
- Transformer or RNN superiority;
- language-model-scale usefulness;
- biological relevance.

The experiment is designed to identify reproducible anomalies and mechanistic candidates for later focused confirmation.

A failed explanation is not the same as a failed phenomenon.
