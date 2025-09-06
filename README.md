# Attractor Curve Mapping Theory

![Visualization](visualization.gif)

## Core Theory

Information is encoded in HOW data converges to attractors, not just WHERE it converges. Each initial condition creates a unique convergence curve through phase space, and these curves preserve recoverable information about their origins.

## Key Principle

When data points converge in a dynamical system:
1. Each point follows a unique trajectory based on its properties
2. The curve shape (not just destination) encodes information
3. Different initial values create distinguishably different curves
4. Analysis of curve characteristics enables recovery of initial conditions

## The Insight

Traditional view: Attractors destroy information (many inputs → one output)

This work shows: The convergence curve preserves information through:
- Path taken through phase space
- Velocity profile during convergence  
- Sequence of dominant attractors
- Oscillation patterns
- Total trajectory length

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
2025 Zachary Daniels
```

## Future Directions

- Application to real dynamical systems
- Hardware implementations using analog circuits
- Integration with reservoir computing
- Analysis of natural systems with attractors

## License

MIT - Open for exploration and extension
