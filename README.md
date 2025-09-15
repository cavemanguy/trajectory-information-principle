 # Attractor Curve Mapping Theory

![Visualization](visualization.gif)

## 
In this recreational research project, I explore an intriguing phenomenon in dynamical systems: how initial data "chooses" between multiple stable attractors
based on intrinsic compatibility rather than mere proximity. While tinkering with convergence dynamics, I noticed that data points exhibit selective affinity for certain attractors - a resonance that goes beyond simple distance metrics in phase space. This observation led me to implement a system where each trajectory becomes a unique signature encoding both the initial conditions and the attractor landscape it navigates.

The core insight emerged from watching how different input values, even when numerically close, would sometimes diverge to entirely different stable states.
Rather than treating this as simple basin boundary crossing, I began exploring whether the data itself carries properties that create preferential "resonance"
with specific attractors. My implementation demonstrates this through a compatibility function that allows attractors to selectively capture data based on
intrinsic signatures. Each convergence trajectory thus becomes a rich encoding of how that particular data pattern interacts with the system's structure - a joint signature of both the information and the computational landscape.

What makes this exploration particularly fascinating is that the trajectory serves triple duty: it encodes the original data, reveals the structure of the attractor landscape, and captures their interaction dynamics. This suggests that stable attractors could serve as computational primitives where the selection pattern itself performs computation - a form of morphological computation where the phase space geometry does the work. While established fields like reservoir computing use transient dynamics, this specific angle of selective attractor affinity based on data-attractor resonance appears less explored. The trajectories become a kind of dynamical hash function, where similar inputs might deliberately diverge to different attractors based on subtle property differences, creating a natural classification or encoding scheme.

This work is purely exploratory - a curiosity-driven investigation into whether these dynamics could offer new perspectives on information processing in complex systems. The potential applications range from novel encoding schemes to pattern classification systems that leverage natural dynamics rather than forced computation.

## Core Theory

Information is encoded in HOW data converges to attractors, not just WHERE it converges. Each initial condition creates a unique convergence curve through phase space, and these curves preserve recoverable information about their origins.

## Key Principle

When data points converge in a dynamical system:
1. Each point follows a unique trajectory based on its properties
2. The curve shape (not just destination) encodes information
3. Different initial values create distinguishably different curves
4. Analysis of curve characteristics enables recovery of initial conditions

## The Key Distinction

**Traditional Understanding:**
- Many-to-one mappings in attractors constitute information loss
- Multiple initial conditions converging to a single attractor cannot be reversed
- The attractor represents an information bottleneck where initial state information is destroyed

**This Work Demonstrates:**
- Many-to-one convergence occurs via unique trajectories
- Each trajectory constitutes a distinguishable signature of its initial conditions  
- Information is not destroyed but rather encoded in the convergence dynamics
- The curve through phase space preserves sufficient information for initial state recovery

**Relationship to Existing Work:**

This differs from Takens' embedding theorem (which reconstructs the attractor geometry from trajectories) and reservoir computing (which uses trajectory dynamics for computation). Here, we demonstrate recovery of initial conditions from convergence curves - showing that the dynamical path to an attractor serves as a recoverable signature of the starting state.

## Quick Demo

```python
from attractor_curve_mapping import AttractorMapper

# Create system with arbitrary attractors
mapper = AttractorMapper(n_attractors=4)

# Different values create different curves
curve1 = mapper.converge(42)
curve2 = mapper.converge(43)

# Curves are unique and recoverable
signature1 = mapper.extract_curve_signature(curve1)
signature2 = mapper.extract_curve_signature(curve2)

# Recovery from curve alone
candidates = mapper.recover_from_curve(curve1)
print(f"Recovered: {candidates[0][0]}")  # Returns 42
```

## Files

- `attractor_curve_mapping.py` - Core theory implementation
- `computational_primitive_demo.py` - Computational applications
- `comparison_traditional.py` - Comparison with traditional methods

## Applications

### Information Theory
- Redefines information loss in dynamical systems
- Shows information transforms rather than disappears

### Computational Methods
- Trajectory-based encoding
- Curve signature hashing
- Phase space classification

### Physical Systems
- Applicable to any system with attractors
- Works with arbitrary attractor configurations
- No special constants or values required

## Mathematical Foundation

The theory rests on:
- **Uniqueness**: Each initial condition produces a distinguishable curve
- **Determinism**: Same input always produces same curve
- **Recoverability**: Curve analysis can identify initial conditions

## Not Claims Being Made

- Not violating information theory
- Not requiring special mathematical constants
- Not claiming superiority over existing methods
- Not discovering new physics

## What This IS

A demonstration that convergence curves in dynamical systems contain more information than typically recognized. The trajectory itself is a rich signature of the initial conditions.

## Installation

```bash
pip install numpy matplotlib scipy
```

## Run Demonstrations

```bash
# See the theory in action
python attractor_curve_mapping.py

# Computational applications
python computational_primitive_demo.py

# Performance comparison
python comparison_traditional.py
```

## Key Results

- 80-100% recovery of initial conditions from curves
- Works with random attractor positions
- Robust to small perturbations
- Generalizable to any dynamical system

## Citation

```
Attractor Curve Mapping: Information Preservation in Convergence Dynamics
Demonstrates that convergence trajectories encode recoverable information
2024 Zachary Daniels, Luke Beard, Ryan Longmire
```

## Future Directions

- Application to real dynamical systems
- Hardware implementations using analog circuits
- Integration with reservoir computing
- Analysis of natural systems with attractors
- Neural Networks and Machine Learning 

## License

MIT - Open for exploration and extension
