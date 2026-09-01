# Mechanism and Architecture

This document defines the current Active Latent Interrogation mechanism. It separates the intuitive story from the testable computational claim.

## 1. Research question

### Plain English

Can a model retrieve a requested fact from a fixed internal memory by choosing how to gently disturb it and then observing the response? More specifically, does the query help choose a useful disturbance direction?

### Technical formulation

Given memory input $x$, query $q$, and target $y$, learn a probe policy and readout that minimize

$$\mathcal L=\mathbb E_{(x,q,y)}[\ell(R(r,q),y)],$$

where response $r$ is induced by a controlled perturbation of a frozen latent-memory system. The scientific comparison is whether performance survives capacity-matched controls and depends specifically on query-directed intervention.

## 2. Frozen compressed latent memory

### Plain English

Several relations are packed into one latent vector. During the probe experiment, the memory is held fixed so the probe cannot rewrite it to make retrieval easier.

### Technical formulation

An encoder produces

$$m=E_\phi(x), \qquad m\in\mathbb R^{d_m}.$$

*Frozen memory* means the relevant memory parameters $\phi$ and response-dynamics parameters are not updated during probe-policy/readout training. *Compressed* is architectural unless measured. A defensible compression claim requires input information, latent rate or dimension, and reconstruction or task distortion.

Freezing parameters does not prevent leakage through activations. Every component's input access must be stated explicitly.

## 3. Learned probe direction

### Plain English

The policy looks at the query—and, currently, the memory—and chooses which direction to push. Push size is controlled separately so improvement cannot come only from a larger disturbance.

### Technical formulation

The current policy is

$$\tilde v=P_\theta(h,m,q), \qquad v=\frac{\tilde v}{\lVert\tilde v\rVert_2+\varepsilon}.$$

The intervention is

$$\delta=\alpha v, \qquad m^+=m+\delta.$$

Normalizing $v$ separates direction from magnitude. Experiments must fix $\alpha$ or report how it is chosen and constrain it identically across controls. If $h$ contains memory-derived state, that access counts as part of $P(h,m,q)$.

## 4. Dynamical response

### Plain English

The decoder is not handed the original memory. The system is nudged and the resulting change is recorded. That change is the response used for retrieval.

### Technical formulation

Let $F$ be the frozen response operator. A minimal finite-difference response is

$$r_1=F(m+\alpha v)-F(m).$$

For dynamics $z_{t+1}=f(z_t)$ with $z_0=m$, a multi-step response may be

$$r_{0:T}=\rho(\{z_t^+-z_t\}_{t=0}^{T}),$$

where $z_0^+=m+\alpha v$ and $\rho$ is a declared trajectory summarizer. For small perturbations,

$$F(m+\alpha v)-F(m)\approx\alpha J_F(m)v.$$

This local approximation must not be assumed for large $\alpha$ or strongly nonlinear responses. The implementation must state whether $r$ includes final differences, full trajectories, hidden states, norms, or learned summaries; these are different information channels.

## 5. Retrieval readout

### Plain English

The decoder gets the query and response. It should not secretly receive the original memory, selected probe vector, or unperturbed hidden states unless that access is a named control.

### Technical formulation

The intended restricted readout is

$$\hat y=R_\psi(r,q).$$

For the main ALI condition, prohibit direct access to $m$, $h$, $v$, $x$, or unreported states. If the decoder receives $m$, the experiment no longer tests retrieval through the induced response alone.

## 6. Current and next addressing policies

### Plain English

Right now the probe chooser can inspect memory. It may be computing part of retrieval while constructing the probe rather than using the query as an independent address. The next control lets it see only the query.

### Technical formulation

Current:

$$v=P(h,m,q).$$

Critical next control:

$$v_q=P_q(q),\quad r_q=F(m+\alpha v_q)-F(m),\quad \hat y_q=R_q(r_q,q).$$

The readout remains restricted to $(r_q,q)$. Comparing these estimates the value of content-dependent probe selection. It does not prove the remaining query-only policy implements symbolic addressing.

## 7. Required controls

### Plain English

A good score is not enough. Compare probing against simpler, wider, query-blind, wrong-direction, and no-perturbation alternatives.

### Technical controls

1. Direct frozen-state readout: $\hat y=R_m(m,q)$.
2. Capacity-matched or wider direct readout with declared parameters and compute.
3. Query-blind probing: $v=P(h,m)$; readout still sees $(r,q)$.
4. Query-only probing: $v=P(q)$; readout still sees $(r,q)$.
5. Wrong-query direction: use $q'\ne q$ only for selecting $v$.
6. Shuffled, mean, and random norm-matched directions.
7. Zero perturbation: $\alpha=0$.
8. Magnitude sweep with direction policy fixed.

All learned controls require comparable tuning budgets, stopping rules, and evaluation splits.

## 8. Limitations of the mechanism claim

### Plain English

Success on a synthetic task would not mean trajectories universally preserve information or that the method scales to language.

### Technical limitations

- A learned perturbation-response pipeline is still ordinary differentiable computation; dynamical language does not establish a distinct primitive.
- Finite differences may expose local sensitivity without stable attractors, convergence, or long-term trajectory coding.
- Because $P(h,m,q)$ sees memory, $v$ is a potential side channel unless its dimension, precision, norm, and decoder access are controlled.
- Synthetic retrieval does not establish naturalistic memory performance or distribution-shift robustness.
- Single point estimates cannot establish superiority.
