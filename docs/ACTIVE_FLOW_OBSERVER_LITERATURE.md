# Active Flow Observer — Literature Review and Claim Boundaries

Reviewed 2026-09-07. The six numbered entries preserve the user's supplied URLs. Source-derived findings below are distinguished from proposed uses in this project. The papers do not jointly demonstrate an exact, continuous semantic interpreter of an arbitrary black box. Source 1 could not be resolved reliably, and source 3 was resolved through independent bibliographic evidence. Do not silently replace missing sources with related papers.

## [1] NeurIPS 2025 poster 117542 — unresolved

Original: https://neurips.cc/virtual/2025/poster/117542

The supplied poster endpoint could not be fetched, and exact-ID searches did not establish its title, authors, or content. **Status: unverified; no scientific claim is attributed to this reference.** A title, PDF, or corrected link is needed before it can be used as evidence. The separately identified Flow Equivariant Recurrent Neural Networks paper is listed below as additional context, not as a substitute for this unknown poster.

## [2] Block-Recurrent Dynamics in ViTs

Original: https://arxiv.org/html/2512.19941v6

Mozes Jacobs, Thomas Fel, Richard Hakim, Alessandra Brondetta, Demba Ba, and T. Andy Keller. arXiv:2512.19941v6, 17 March 2026. The paper introduces the Block-Recurrent Hypothesis and Raptor, a recurrent surrogate intended to reproduce intermediate activations of trained Vision Transformers. It uses representational phase segmentation and teacher-forced/autoregressive reconstruction. Its reported DINOv2 results recover 96% of teacher linear-probe accuracy with two recurrent blocks and 98% with three. It also studies directional convergence, perturbation correction, token-specific motion, and low-rank late-depth updates.

**Project use (inference):** intermediate-trajectory reconstruction provides a useful observer/surrogate baseline; angular convergence, sensitivity, and low-rank analyses can be compared with R8/AD6 without assuming the same mechanism. Layerwise functional fidelity and final-task fidelity should remain separate. The paper does not establish real-time semantic decoding, arbitrary-model block recurrence, or information beyond complete state-plus-map. It is relevant prior art to any claim that recurrent approximations or dynamical analysis of Transformers are novel.

## [3] Block-Recurrent Dynamics in Vision Transformers — same work as [2]

Original: https://openreview.net/forum?id=gH3HhnfWLC

The OpenReview page was blocked by browser verification. The exact forum ID is identified in the authors' publication/CV material and in an independent paper's bibliography as the ICLR 2026 version of Block-Recurrent Dynamics in Vision Transformers. The matching author list and title support treating it as the same Raptor work as [2], not a sixth independent result. The verified arXiv version remains the basis for the technical summary. We have not independently inspected OpenReview reviews or camera-ready differences.

Corroborating bibliographic sources:
- https://akandykeller.github.io/assets/img/about_me/cv.pdf
- https://openreview.net/pdf/d0fa2761173eee3008b77739909ed53e66b2e83b.pdf

## [4] A Robust and Explainable Transformer-Based Framework for Phishing Email Detection

Original: https://arxiv.org/html/2511.12085v2

Sajad U P, arXiv:2511.12085v2, 6 February 2026. This work combines DistilBERT phishing classification, embedding-level Fast Gradient Method training, character-level perturbation, LIME/SHAP/Integrated Gradients attribution, and Flan-T5-Small plain-language explanations guided by structured evidence. The paper reports improvements over its standard DistilBERT baseline and includes explanation-stability and user-centered evaluation.

**Project use (inference):** an evidence-structured explanation layer, robustness tests, and separate human-readable presentation are useful design ideas. This is a phishing-classification/XAI study, not a demonstration of hidden algorithm recovery. Attribution and fluent explanation should not be equated with causal faithfulness. Its generated text cannot serve as ground truth for our semantic observer.

## [5] Probing Latent Subspaces of LLMs for AI Security: Identifying and Manipulating Adversarial States

Original: https://arxiv.org/html/2503.09066v2

arXiv:2503.09066v2, 4 July 2025. The study extracts LLM activations, uses Linear Discriminant Analysis to separate safe and jailbreak-associated representations, derives a direction between those groups, and intervenes in hidden states. It reports statistically significant jailbreak-response changes on a subset of prompts, changes in downstream activations, and layer-specific effects. Its primary setting uses Llama-3.1-8B-Instruct, with a smaller Qwen comparison. The paper's labels and experimental context constrain the interpretation.

**Project use (inference):** explicit activation hooks, targeted directions, and downstream causal measurement are relevant. A learned class-separating vector is not necessarily a native flow tangent, a semantic operation, or a universal control axis. We should use benign, authorized synthetic interventions rather than adopt jailbreak induction as the project task. Direction-only, target-label, and controller-payload leakage must be audited separately. The paper does not establish exact internal logic reconstruction.

## [6] In-Context System Identification for Nonlinear Dynamics Using Large Language Models

Original: https://arxiv.org/pdf/2602.07360

Linyu Lin, arXiv:2602.07360v1, 7 February 2026; six-page preprint submitted to CCTA 2026. The method uses an LLM to propose candidate symbolic structures for a SINDy sparse-regression loop. Candidates are grammar-checked, fitted, simulated, and evaluated against numerical and structural criteria. The paper reports experiments on 63 ODEBench systems and a March–Leuba boiling-reactor model, with improved recovery relative to its baseline for complex dynamics. Some recovered forms are approximate rather than exact.

**Project use (inference):** this is a strong reason to include conventional system identification and a constrained symbolic-hypothesis generator before claiming a special neural semantic decoder. It identifies governing equations of measured dynamical systems, not necessarily the algorithm implemented inside a neural network. The paper's refinement procedure uses test-error feedback in candidate selection; our confirmatory protocol must reserve a genuinely untouched final test set and must not use final-test results to select equations, prompts, or stopping points. Symbolic prediction accuracy and structural identifiability are distinct.

## Additional relevant prior art, not replacements for the six sources

**Flow Equivariant Recurrent Neural Networks**, T. Anderson Keller, NeurIPS 2025 Spotlight. https://arxiv.org/abs/2507.14793 and https://proceedings.nips.cc/paper_files/paper/2025/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html. The paper studies equivariance to time-parameterized symmetry flows, rather than assuming ordinary recurrence respects such transformations. It reports improved length/velocity generalization and training behavior on its sequence tasks. This is relevant if an AFO follow-up tests known flow symmetries, but we must not impose an equivariant architecture on a native-dynamics discovery experiment without a separate hypothesis.

**Transformer Dynamics: A neuroscientific approach to interpretability of large language models**, Jesseba Fernando and Grigori Guitchounts, arXiv:2502.12131. https://arxiv.org/abs/2502.12131. It studies Transformer residual-stream evolution across depth, including continuity and reduced-dimensional curved trajectories. It is direct prior art for treating Transformer depth as a dynamical system. Its descriptive trajectory findings do not establish exact semantic logic recovery or universal attractor behavior.

## Implications for novelty and the next experiment

The individual ingredients—recurrent observers, state-space analysis, targeted activation interventions, recurrent Transformer surrogates, symbolic system identification, and natural-language explanations—have substantial prior art. A new contribution would need to be established by a specific controlled result, such as causally informative active observations improving held-out operation recovery beyond full-state, passive-history, controller-only, and ordinary system-identification baselines under strict information constraints. An architecture diagram alone is not a novelty claim. The [preparation protocol](ACTIVE_FLOW_OBSERVER.md) defines the tests needed to distinguish such a result from answer transmission, endpoint prediction, descriptive geometry, and plausible but unfaithful text.