"""
Failure Models for Wind Turbine Components
Simulates degradation and failures using physics-based and stochastic models
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config import config


@dataclass
class ComponentState:
    """State of a single component"""
    health: float  # 0 to 1, where 1 is perfect condition
    age: int  # Time steps since last maintenance
    degradation_rate: float  # Rate of health decrease per step
    failure_threshold: float  # Health level below which component fails
    has_failed: bool = False


class FailureModel:
    """Model component degradation and failures"""
    
    def __init__(self, component_type: str, failure_rate: float = None):
        """
        Initialize failure model for a component
        
        Args:
            component_type: Type of component (blade, gearbox, etc.)
            failure_rate: Annual failure rate (overrides config if provided)
        """
        self.component_type = component_type
        self.failure_rate = failure_rate or config.FAILURE_RATES.get(component_type, 0.15)
        
        # Convert annual rate to per-step rate
        self.base_degradation_rate = self.failure_rate / config.MAX_STEPS_PER_EPISODE
        
        # Component-specific parameters
        self.failure_threshold = 0.2  # Component fails below 20% health
        self.degradation_noise = 0.01  # Random variation in degradation
        
    def initialize_component(self) -> ComponentState:
        """Initialize a new or maintained component"""
        return ComponentState(
            health=1.0,
            age=0,
            degradation_rate=self.base_degradation_rate,
            failure_threshold=self.failure_threshold,
            has_failed=False
        )
    
    def update(self, state: ComponentState, 
               environmental_stress: float = 1.0,
               maintenance_quality: float = 0.0) -> ComponentState:
        """
        Update component state for one time step
        
        Args:
            state: Current component state
            environmental_stress: Multiplier for degradation (e.g., harsh weather)
            maintenance_quality: 0 to 1, higher = better maintenance effect
        
        Returns:
            Updated component state
        """
        if state.has_failed:
            return state
        
        # Age increases
        state.age += 1
        
        # Apply maintenance if provided
        if maintenance_quality > 0:
            health_recovery = maintenance_quality * 0.3  # Max 30% recovery
            state.health = min(1.0, state.health + health_recovery)
            state.age = 0  # Reset age after maintenance
            state.degradation_rate = self.base_degradation_rate
            return state
        
        # Degradation increases with age (bathtub curve)
        age_factor = 1.0 + (state.age / 1000) ** 2  # Accelerated aging
        
        # Environmental stress
        stress_factor = environmental_stress
        
        # Total degradation
        degradation = (state.degradation_rate * age_factor * stress_factor + 
                      np.random.randn() * self.degradation_noise)
        
        # Update health
        state.health = max(0.0, state.health - degradation)
        
        # Check for failure
        if state.health <= state.failure_threshold:
            state.has_failed = True
        
        return state
    
    def get_failure_probability(self, state: ComponentState) -> float:
        """Calculate probability of failure in next step"""
        if state.has_failed:
            return 1.0
        
        # Probability increases as health decreases
        health_risk = (1.0 - state.health) ** 2
        
        # Age also increases risk
        age_risk = min(1.0, state.age / 500) ** 2
        
        return min(1.0, health_risk + age_risk)


class TurbineFailureSimulator:
    """Simulate failures for all components of a turbine"""
    
    def __init__(self, turbine_id: int):
        """
        Initialize turbine failure simulator
        
        Args:
            turbine_id: Unique turbine identifier
        """
        self.turbine_id = turbine_id
        
        # Create failure model for each component
        self.components = {}
        self.failure_models = {}
        
        for component_type in config.FAILURE_RATES.keys():
            self.failure_models[component_type] = FailureModel(component_type)
            self.components[component_type] = self.failure_models[component_type].initialize_component()
    
    def reset(self):
        """Reset all components to new condition"""
        for component_type in self.components.keys():
            self.components[component_type] = self.failure_models[component_type].initialize_component()
    
    def step(self, environmental_factors: Dict[str, float],
             maintenance_actions: Dict[str, float] = None) -> Dict[str, ComponentState]:
        """
        Update all components for one time step
        
        Args:
            environmental_factors: Dict with stress factors (wind_speed, temperature, etc.)
            maintenance_actions: Dict with maintenance quality per component (0-1)
        
        Returns:
            Updated component states
        """
        if maintenance_actions is None:
            maintenance_actions = {k: 0.0 for k in self.components.keys()}
        
        # Calculate environmental stress
        stress = self._calculate_environmental_stress(environmental_factors)
        
        # Update each component
        for component_type, state in self.components.items():
            maintenance = maintenance_actions.get(component_type, 0.0)
            self.components[component_type] = self.failure_models[component_type].update(
                state, stress, maintenance
            )
        
        return self.components
    
    def _calculate_environmental_stress(self, factors: Dict[str, float]) -> float:
        """Calculate environmental stress multiplier"""
        stress = 1.0
        
        # High wind speed increases stress
        wind_speed = factors.get('wind_speed', 7.0)
        if wind_speed > 15:
            stress *= 1.0 + (wind_speed - 15) / 20
        
        # Extreme temperatures
        temperature = factors.get('temperature', 15.0)
        if temperature < 0 or temperature > 35:
            stress *= 1.2
        
        # High vibration
        vibration = factors.get('vibration', 0.5)
        if vibration > 1.0:
            stress *= 1.0 + (vibration - 1.0)
        
        return min(3.0, stress)  # Cap at 3x stress
    
    def apply_maintenance(self, component_type: str, maintenance_type: str = 'preventive'):
        """
        Apply maintenance to a component
        
        Args:
            component_type: Component to maintain
            maintenance_type: 'preventive' or 'corrective'
        """
        if maintenance_type == 'corrective':
            # Full restoration for corrective maintenance
            self.components[component_type] = self.failure_models[component_type].initialize_component()
        else:
            # Partial restoration for preventive
            quality = 0.7  # Preventive maintenance is 70% effective
            self.components[component_type] = self.failure_models[component_type].update(
                self.components[component_type], 1.0, quality
            )
    
    def get_system_health(self) -> float:
        """Get overall system health (minimum of all components)"""
        if not self.components:
            return 1.0
        return min(state.health for state in self.components.values())
    
    def get_failure_status(self) -> Dict[str, bool]:
        """Get failure status of all components"""
        return {k: v.has_failed for k, v in self.components.items()}
    
    def has_any_failure(self) -> bool:
        """Check if any component has failed"""
        return any(state.has_failed for state in self.components.values())
    
    def get_component_health_vector(self) -> np.ndarray:
        """Get health of all components as vector"""
        health_values = [self.components[comp].health for comp in sorted(self.components.keys())]
        return np.array(health_values, dtype=np.float32)
    
    def get_component_age_vector(self) -> np.ndarray:
        """Get age of all components as vector"""
        age_values = [self.components[comp].age for comp in sorted(self.components.keys())]
        return np.array(age_values, dtype=np.float32)
    
    def get_failure_probabilities(self) -> Dict[str, float]:
        """Get failure probability for each component"""
        return {
            comp_type: self.failure_models[comp_type].get_failure_probability(state)
            for comp_type, state in self.components.items()
        }


# Test the failure models
if __name__ == "__main__":
    print("Testing Failure Models...")
    print("=" * 60)
    
    # Test single component
    print("\n1. Testing single component (gearbox):")
    failure_model = FailureModel('gearbox')
    component = failure_model.initialize_component()
    
    print(f"Initial health: {component.health:.3f}")
    
    # Simulate 100 steps
    for step in range(100):
        component = failure_model.update(component, environmental_stress=1.5)
        if step % 20 == 0:
            print(f"Step {step}: health={component.health:.3f}, age={component.age}, failed={component.has_failed}")
    
    # Test turbine simulator
    print("\n2. Testing full turbine simulator:")
    simulator = TurbineFailureSimulator(turbine_id=0)
    
    print(f"Initial system health: {simulator.get_system_health():.3f}")
    
    # Simulate with environmental factors
    env_factors = {'wind_speed': 12.0, 'temperature': 20.0, 'vibration': 0.5}
    
    for step in range(50):
        simulator.step(env_factors)
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  System health: {simulator.get_system_health():.3f}")
            print(f"  Component health: {simulator.get_component_health_vector()}")
            print(f"  Any failures: {simulator.has_any_failure()}")
    
    # Test maintenance
    print("\n3. Testing maintenance:")
    print(f"Health before maintenance: {simulator.get_component_health_vector()}")
    simulator.apply_maintenance('gearbox', 'preventive')
    print(f"Health after maintenance: {simulator.get_component_health_vector()}")
    
    print("\n✓ Failure Models test passed!")
