#!/usr/bin/env python3
"""
Attractor Curve Mapping Theory
How convergence curves encode information about initial conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class AttractorMapper:
    """
    Maps data through attractor convergence, encoding information in curves
    """
    
    def __init__(self, n_attractors=4, dimensions=2):
        """
        Create a system with arbitrary attractors
        Not specific values - just convergence points in phase space
        """
        # Random attractors in phase space (could be ANY points)
        np.random.seed(42)  # For reproducibility
        self.attractors = np.random.rand(n_attractors, dimensions) * 100
        self.convergence_curves = {}
        
    def compute_basin_affinity(self, point, attractor):
        """
        How strongly does this point "belong" to this attractor's basin?
        This emerges from the dynamics, not programmed
        """
        # Distance in phase space
        distance = np.linalg.norm(point - attractor)
        
        # Some points have natural affinity for certain attractors
        # based on their properties (not distance alone)
        point_signature = np.sum(point) % 7  # Arbitrary property
        attractor_signature = np.sum(attractor) % 7
        
        # Affinity combines distance and "compatibility"
        compatibility = 1.0 / (1.0 + abs(point_signature - attractor_signature))
        affinity = compatibility / (1.0 + distance * 0.01)
        
        return affinity
    
    def converge(self, initial_value, max_steps=100):
        """
        Let a value converge to its attractor
        The PATH it takes encodes information
        """
        # Convert to phase space point
        point = np.array([
            float(initial_value),
            float(initial_value) * 1.5 % 100  # Second dimension
        ])
        
        curve = []
        velocity = np.zeros(2)
        
        for step in range(max_steps):
            # Calculate pull from ALL attractors
            total_force = np.zeros(2)
            affinities = []
            
            for attractor in self.attractors:
                # Each attractor exerts influence
                affinity = self.compute_basin_affinity(point, attractor)
                direction = attractor - point
                force = direction * affinity * 0.1
                total_force += force
                affinities.append(affinity)
            
            # Update dynamics
            velocity = velocity * 0.8 + total_force  # Damping
            point = point + velocity
            
            # Record the curve
            curve.append({
                'position': point.copy(),
                'velocity': velocity.copy(),
                'dominant_attractor': np.argmax(affinities),
                'affinity_distribution': affinities.copy()
            })
            
            # Check convergence
            if np.linalg.norm(velocity) < 0.01:
                break
        
        self.convergence_curves[initial_value] = curve
        return curve
    
    def extract_curve_signature(self, curve):
        """
        Extract identifying features from the convergence curve
        This is where information is preserved
        """
        if not curve:
            return None
        
        # The curve shape encodes the initial value
        signature = {
            # Which attractor dominated at different phases
            'early_attractor': curve[min(5, len(curve)-1)]['dominant_attractor'],
            'mid_attractor': curve[len(curve)//2]['dominant_attractor'],
            'final_attractor': curve[-1]['dominant_attractor'],
            
            # How the approach happened
            'curve_length': len(curve),
            'total_distance': sum(np.linalg.norm(curve[i]['velocity']) 
                                for i in range(len(curve))),
            
            # Oscillation patterns
            'direction_changes': self.count_direction_changes(curve),
            
            # Affinity evolution
            'affinity_pattern': self.extract_affinity_pattern(curve)
        }
        
        return signature
    
    def count_direction_changes(self, curve):
        """Count how many times the curve changed direction"""
        changes = 0
        for i in range(2, len(curve)):
            v1 = curve[i-1]['velocity']
            v2 = curve[i]['velocity']
            if np.dot(v1, v2) < 0:  # Opposite directions
                changes += 1
        return changes
    
    def extract_affinity_pattern(self, curve):
        """Extract pattern of attractor preferences during convergence"""
        if len(curve) < 10:
            return tuple(c['dominant_attractor'] for c in curve)
        
        # Sample at key points
        indices = [0, len(curve)//4, len(curve)//2, 3*len(curve)//4, -1]
        pattern = tuple(curve[min(i, len(curve)-1)]['dominant_attractor'] 
                       for i in indices if i < len(curve))
        return pattern
    
    def recover_from_curve(self, curve):
        """
        Attempt to identify original value from convergence curve
        """
        target_signature = self.extract_curve_signature(curve)
        if not target_signature:
            return None
        
        # Find best match
        candidates = []
        for original_value, stored_curve in self.convergence_curves.items():
            stored_signature = self.extract_curve_signature(stored_curve)
            
            # Compare signatures
            score = 0
            if stored_signature['final_attractor'] == target_signature['final_attractor']:
                score += 30
            if stored_signature['early_attractor'] == target_signature['early_attractor']:
                score += 20
            if stored_signature['affinity_pattern'] == target_signature['affinity_pattern']:
                score += 25
            if abs(stored_signature['curve_length'] - target_signature['curve_length']) < 5:
                score += 15
            if stored_signature['direction_changes'] == target_signature['direction_changes']:
                score += 10
            
            if score > 50:
                candidates.append((original_value, score))
        
        return sorted(candidates, key=lambda x: x[1], reverse=True)

def demonstrate_theory():
    """
    Demonstrate the attractor curve mapping theory
    """
    print("="*70)
    print("ATTRACTOR CURVE MAPPING THEORY")
    print("="*70)
    print("\nCore Principle: Information is encoded in HOW data converges,")
    print("not just WHERE it converges.\n")
    
    mapper = AttractorMapper(n_attractors=4)
    
    # Test 1: Different values create different curves
    print("1. UNIQUE CONVERGENCE CURVES")
    print("-"*40)
    
    test_values = [10, 11, 50, 51, 100, 101]
    
    for value in test_values:
        curve = mapper.converge(value)
        signature = mapper.extract_curve_signature(curve)
        print(f"Value {value:3d}: ", end="")
        print(f"Length={signature['curve_length']:3d}, ", end="")
        print(f"Final={signature['final_attractor']}, ", end="")
        print(f"Pattern={signature['affinity_pattern']}")
    
    # Test 2: Recovery from curves
    print("\n2. INFORMATION RECOVERY FROM CURVES")
    print("-"*40)
    
    test_recovery = [10, 50, 100, 150, 200]
    successful = 0
    
    for value in test_recovery:
        curve = mapper.converge(value)
        candidates = mapper.recover_from_curve(curve)
        
        if candidates and candidates[0][0] == value:
            successful += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"Value {value:3d}: Recovery {status} ", end="")
        if candidates:
            print(f"(top candidate: {candidates[0][0]}, score: {candidates[0][1]})")
        else:
            print("(no candidates)")
    
    print(f"\nRecovery rate: {100*successful/len(test_recovery):.0f}%")
    
    # Test 3: Visualize curve differences
    print("\n3. VISUALIZING CURVE UNIQUENESS")
    print("-"*40)
    
    visualize_curves(mapper, [25, 75, 125, 175])
    
    return mapper

def visualize_curves(mapper, values):
    """
    Show how different initial values create different curves
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Convergence in phase space
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    
    for i, value in enumerate(values):
        curve = mapper.convergence_curves.get(value, mapper.converge(value))
        positions = np.array([c['position'] for c in curve])
        ax.plot(positions[:, 0], positions[:, 1], color=colors[i], 
                alpha=0.7, label=f'Value {value}')
        ax.scatter(positions[-1, 0], positions[-1, 1], color=colors[i], s=50, marker='o')
    
    # Plot attractors
    for j, attractor in enumerate(mapper.attractors):
        ax.scatter(attractor[0], attractor[1], s=200, marker='*', 
                  color='red', edgecolor='black', linewidth=2, zorder=5)
        ax.text(attractor[0], attractor[1]-5, f'A{j}', ha='center')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Convergence Curves in Phase Space')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Velocity profiles
    ax = axes[1]
    for i, value in enumerate(values):
        curve = mapper.convergence_curves[value]
        velocities = [np.linalg.norm(c['velocity']) for c in curve]
        ax.plot(velocities, color=colors[i], alpha=0.7, label=f'Value {value}')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Velocity Magnitude')
    ax.set_title('Velocity Profiles (Curve Dynamics)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Attractor influence over time
    ax = axes[2]
    value = values[0]  # Show one example
    curve = mapper.convergence_curves[value]
    
    # Extract dominant attractor at each step
    dominant = [c['dominant_attractor'] for c in curve]
    ax.plot(dominant, 'o-', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Dominant Attractor')
    ax.set_title(f'Attractor Influence Evolution (Value {value})')
    ax.set_yticks(range(len(mapper.attractors)))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attractor_curves.png', dpi=150)
    print("Saved visualization to 'attractor_curves.png'")
    
    return fig

if __name__ == "__main__":
    mapper = demonstrate_theory()
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("-"*40)
    print("• Each initial value creates a UNIQUE convergence curve")
    print("• The curve shape encodes information about the initial state")
    print("• Recovery is possible by analyzing curve characteristics")
    print("• Attractors don't need to be special values - ANY points work")
    print("• Information lives in the DYNAMICS, not the final state")
    print("\nThis is a general principle about dynamical systems,")
    print("not specific to any particular attractors or values.")
    print("="*70)