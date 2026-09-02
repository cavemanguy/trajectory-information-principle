# G1 protocol/analysis note

The first implementation of the largest-Lyapunov estimator perturbed only part of the true Markov state and incorrectly reset the previous-h component used by q_i and tangential-speed calculations. This produced impossible positive estimates even at α=0, where the known base map is h→0.92h.

The issue was detected by the α=0 sanity check before final interpretation. The estimator was replaced by a twin-trajectory calculation over the complete dynamical state [h_t, h_{t-1}, o_1,...,o_S], with renormalization in that full state. The corrected α=0 estimate is about log(0.92)=-0.08338, as expected.

All reported final Lyapunov values, regime classifications, the report, evidence table, and summary use the corrected estimator. The underlying trajectories, observer equations, alpha sweep, geometry, and interventions were not altered.
