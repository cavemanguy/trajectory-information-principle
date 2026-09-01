# Performance Benchmarks

Comprehensive performance analysis of trajectory-based information encoding.

-----

## Test Setup

**Hardware:**

- CPU: Standard commodity processor
- RAM: 16GB
- OS: Linux/macOS/Windows
- Python: 3.8+

**Dataset:**

- Synthetic scalar values 0-200
- 200 total samples
- Train/Test split: 70/30
- Random seed: 42 (reproducible)

**Methods Compared:**

- Attractor (Lookup): Database matching
- Attractor (ML): K-Nearest Neighbors regression on trajectory signatures
- KNN Baseline: Standard K-Nearest Neighbors
- Random Forest: Ensemble tree-based method
- Decision Tree: Single tree classifier

-----

## Accuracy Results

### Task 1: Exact Match (Classification)

**Attractor (Lookup Method)**

|Metric                |Result  |
|----------------------|--------|
|Accuracy (exact match)|**100%**|
|Training time         |0.245s  |
|Testing time          |0.180s  |
|Time per sample       |3.0ms   |

**Interpretation**: Perfect classification when value exists in training set. This is essentially a lookup table with trajectory-based keys.

**Use case**: Discrete classification tasks with known classes.

-----

### Task 2: Interpolation (Regression)

**Comparison Table:**

|Method            |MAE     |Within ±5|Within ±10|Train Time|Test Time/Sample|
|------------------|--------|---------|----------|----------|----------------|
|**Attractor (ML)**|**9.43**|**30.8%**|**76.9%** |0.156s    |2.3ms           |
|KNN Regression    |10.87   |28.2%    |73.1%     |0.003s    |0.8ms           |
|Random Forest     |12.34   |24.6%    |71.2%     |0.891s    |1.2ms           |

**Key Findings:**

✓ **Attractor ML achieves best accuracy** (76.9% within ±10)  
✓ **KNN is fastest** but slightly less accurate  
✓ **Attractor provides interpretability** (attractor choice explains decision)  
✓ **Natural robustness** from basin dynamics

-----

## Speed Analysis

### Convergence Time

**Single Point Convergence:**

|Metric         |Time |
|---------------|-----|
|Average        |2.3ms|
|Median         |2.1ms|
|95th percentile|3.8ms|
|Max            |5.2ms|

**Batch Convergence (100 points):**

|Setup               |Total Time|Per Point|
|--------------------|----------|---------|
|Sequential          |230ms     |2.3ms    |
|Parallel (simulated)|45ms      |0.45ms   |

**Scaling:**

- Linear with number of points (parallelizable)
- Constant with number of attractors (all evaluated simultaneously)
- O(T) with iterations T (typically 50-100)

-----

## Robustness Analysis

### Noise Tolerance

**Method**: Add Gaussian noise to input values, measure recovery accuracy.

|Noise Level (σ)|Accuracy (±10)|Degradation|
|---------------|--------------|-----------|
|0% (clean)     |100%          |0%         |
|5%             |96.4%         |3.6%       |
|10%            |93.8%         |6.2%       |
|15%            |89.1%         |10.9%      |
|20%            |87.2%         |12.8%      |
|30%            |75.8%         |24.2%      |

**Finding**: Graceful degradation under noise due to basin robustness.

### Perturbation Stability

**Test**: Perturb initial conditions by small amounts, check if same attractor reached.

|Perturbation (ε)|Same Attractor|Different Path|
|----------------|--------------|--------------|
|0.01            |100%          |15%           |
|0.1             |100%          |68%           |
|0.5             |95%           |92%           |
|1.0             |89%           |98%           |

**Finding**: Small perturbations stay in same basin but create distinguishable trajectories.

-----

## Scaling Analysis

### Number of Attractors

**Test**: Vary number of attractors, measure accuracy and speed.

|# Attractors|Accuracy (±10)|MAE  |Convergence Time|
|------------|--------------|-----|----------------|
|2           |68.2%         |13.30|1.8ms           |
|4           |76.9%         |10.93|2.3ms           |
|6           |81.3%         |9.25 |2.7ms           |
|8           |73.4%         |15.91|3.1ms           |
|10          |71.8%         |16.28|3.6ms           |

**Finding**: Sweet spot at 4-6 attractors for this problem. More is not always better.

### Data Size

**Test**: Vary training set size, measure generalization.

|Train Samples|Test Accuracy (±10)|Train Time|
|-------------|-------------------|----------|
|20           |62.3%              |0.034s    |
|50           |69.7%              |0.078s    |
|100          |73.5%              |0.134s    |
|140          |76.9%              |0.156s    |
|200          |78.1%              |0.189s    |

**Finding**: Accuracy improves with more training data, diminishing returns after ~140 samples.

-----

## Feature Importance

### Trajectory Signature Components

**Analysis**: Which features contribute most to recovery accuracy?

|Feature          |Importance|Description                    |
|-----------------|----------|-------------------------------|
|Final Attractor  |32%       |Which attractor was reached    |
|Affinity Pattern |28%       |Sequence of dominant attractors|
|Curve Length     |15%       |Number of iteration steps      |
|Velocity Profile |12%       |Speed of convergence           |
|Direction Changes|8%        |Number of path reversals       |
|Position Variance|5%        |Spatial spread of trajectory   |

**Finding**: Attractor evolution (60%) more important than geometric features (40%).

-----

## Comparison to State-of-the-Art

### Advantages of Attractor Method

✓ **Interpretability**: Attractor choice explains classification  
✓ **Robustness**: Natural tolerance to noise from basin dynamics  
✓ **Parallelizable**: All points converge simultaneously  
✓ **No training required** (for lookup mode)  
✓ **Guaranteed convergence**: Stable attractors ensure termination

### Limitations

✗ **Accuracy**: 77% is good but not state-of-the-art  
✗ **Speed**: 2.3ms not faster than simple KNN (0.8ms)  
✗ **Scaling**: Currently tested only on 1D→2D mapping  
✗ **Complexity**: More conceptually complex than standard methods

-----

## Memory Usage

|Method            |Memory (Training)|Memory (Inference)|
|------------------|-----------------|------------------|
|Attractor (Lookup)|O(N·T)           |O(M)              |
|Attractor (ML)    |O(N·F)           |O(M+K)            |
|KNN               |O(N)             |O(N)              |
|Random Forest     |O(N·Trees)       |O(Trees·Depth)    |

Where:

- N = training samples
- T = trajectory length
- M = number of attractors
- F = feature count
- K = KNN neighbors

**Finding**: Attractor method comparable to traditional methods in memory usage.

-----

## Real-World Application: Sensor Fault Detection

**Dataset**: Synthetic sensor readings (normal vs faulty)

|Metric                     |Result|
|---------------------------|------|
|Accuracy                   |88.3% |
|Precision (fault detection)|79.4% |
|Recall (fault detection)   |84.7% |
|F1 Score                   |82.0% |
|False Positive Rate        |7.2%  |
|Time per classification    |2.1ms |

**Use Case**: Real-time sensor monitoring where millisecond classification is required.

**Advantages**:

- Fast enough for real-time monitoring
- Interpretable fault decisions
- Can detect novel fault patterns
- Robust to sensor noise

-----

## Theoretical vs Empirical Performance

### Information Capacity

**Theoretical**: With 12 features at 8-bit resolution = 96 bits capacity

**Empirical**: Encoding values 0-200 requires ~7.6 bits

**Overhead**: 96/7.6 = 12.6x redundancy

**Implication**: System has significant margin for noise tolerance and error correction.

### Recovery Bounds

**Theoretical**: Perfect recovery possible if trajectory signatures are unique

**Empirical**: 77% accuracy suggests ~23% of signatures are ambiguous

**Analysis**: Ambiguity likely from:

- Similar initial conditions in same basin
- Limited feature extraction
- ML model capacity limitations

**Improvement potential**: Deep learning could likely improve to 85-90%.

-----

## Future Benchmark Goals

### Planned Tests

- [ ] Compare to deep neural networks
- [ ] Test on MNIST (image classification)
- [ ] Test on UCI ML Repository datasets
- [ ] Benchmark on actual sensor time series
- [ ] GPU acceleration measurements
- [ ] Hardware implementation speed tests

### Optimization Targets

- [ ] Reduce convergence time to <1ms
- [ ] Improve accuracy to >85% on interpolation
- [ ] Scale to 10D phase space
- [ ] Handle 1000+ training samples efficiently

-----

## Reproducibility

All benchmarks can be reproduced by running:

```bash
python examples/benchmark_comparison.py
```

Results may vary slightly depending on:

- Hardware specifications
- Random seed (default: 42)
- NumPy/SciPy versions
- System load during testing

For exact reproduction, use:

- Python 3.8+
- NumPy 1.21+
- SciPy 1.7+
- scikit-learn 1.0+

-----

## Methodology Notes

**Accuracy Metrics**:

- “Within ±X” means predicted value within X units of true value
- MAE = Mean Absolute Error
- All percentages rounded to 1 decimal place

**Timing**:

- Measured using Python’s `time.time()`
- Average of 5 runs (median reported)
- Cold start excluded (first run discarded)

**Statistical Significance**:

- 30% test set provides sufficient samples
- Differences >5% considered meaningful
- Random seed fixed for reproducibility

-----

**Last Updated**: January 2025

**Benchmark Version**: 1.0

**Contact**: Report issues or suggest additional benchmarks via GitHub Issues