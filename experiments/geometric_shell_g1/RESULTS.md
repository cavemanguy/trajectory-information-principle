# Geometric-Shell G1 — Results

G1 is independent of Attractor-A2. No local observer contains a strange attractor or any explicitly oscillatory/chaotic subsystem. Each octahedral face uses the same two-state leaky observer and sees only its own local measurements `[d_i, q_i, tangential_speed_i]`. Faces communicate only through the shared 16-D phase state.

## Primary result

The uncoupled base map `h_{t+1}=0.92 h_t` remains a simple stable fixed-point system. With surface coupling enabled, identical observers differentiate from symmetric observer initialization because differently oriented faces experience different local measurements.

The frozen coupling sweep was `alpha = {0, .02, .05, .10, .20, .40, .80}`. After correction of the full-state Lyapunov estimator and a 3000-step long-horizon verification of the slow candidates, every primary condition converges to a fixed point. No sustained periodic, quasiperiodic, or chaotic global attractor was established.

Corrected mean largest-Lyapunov estimates were approximately:

| alpha | lambda_max |
|---:|---:|
| 0.00 | -0.08328 |
| 0.02 | -0.00506 |
| 0.05 | -0.03304 |
| 0.10 | -0.06201 |
| 0.20 | -0.08367 |
| 0.40 | -0.08335 |
| 0.80 | -0.08337 |

The shell nevertheless creates stable displaced equilibria. Mean final radius grows from approximately 0 at alpha=0 to 0.25, 0.625, 1.25, 2.5, 4.13, and 5.30 over the nonzero sweep. Multiple symmetry-related final equilibria occur across initial conditions; at alpha=.10 the 32 frozen initial conditions occupy eight distinct final equilibria under a 1e-5 clustering tolerance.

## Surface differentiation and coordination

Across-surface observer-state and force variance grow strongly with coupling despite exact symmetric observer initialization. Opposite octahedral face pairs become almost perfectly anti-correlated in force output in the alpha=.02 relaxation regime.

This is geometry-conditioned differentiation, not spontaneous breaking of a fully symmetric complete state, because the frozen global initial states are generally asymmetric.

## Causal surface interventions

At alpha=.02, single-face interventions alter the later shared trajectory and other observers. Averaged global-state displacements after intervention were approximately:

| Intervention | Mean displacement |
|---|---:|
| silence one surface | 0.1043 |
| freeze one observer | 0.0047 |
| flip one surface force | 0.1285 |
| swap two observer states | 0.0958 |
| swap two surface orientations | 0.1114 |

Thus the intended indirect coupling path `surface_i -> h -> surface_j` is causally active in this synthetic system.

## Controls

A face-label scramble of the same octahedral normals is dynamically identical to the primary geometry, as expected because adjacency metadata is not explicitly used. Random and orthogonal-like normal sets behave differently, but perturbation energies also differ, so G1 does not establish a uniquely octahedral advantage.

The recurrent observer changes relaxation dynamics strongly compared with the memoryless control, but static shaped surface forces also reproduce the displaced-equilibrium phenomenon. Therefore persistent observer memory is not necessary for the existence of shifted equilibria. Random actuation produces persistent noisy motion rather than the orderly fixed-point convergence of the recurrent shell.

## Supported conclusion

A symmetric distributed system of identical local observer-actuators can differentiate, coordinate, and causally influence one another indirectly through a shared high-dimensional phase state. Under the frozen G1 equations, however, that coupling produces multistable displaced equilibria rather than sustained oscillatory or chaotic global dynamics.

G1 does not establish consciousness, intelligence, information creation, neural emergence, or that geometry is computation.
