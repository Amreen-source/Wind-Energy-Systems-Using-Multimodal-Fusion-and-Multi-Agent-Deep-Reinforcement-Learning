"""
Wind Farm Environment for Multi-Agent Reinforcement Learning
Gymnasium-compatible environment for predictive maintenance
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any

from config import config
from data_pipeline import DataPipeline
from failure_models import TurbineFailureSimulator
from reward_functions import RewardFunction


class WindFarmEnv(gym.Env):
    """
    Multi-agent wind farm maintenance environment
    
    Observation Space (per agent):
        - SCADA data: (sequence_length, num_features)
        - Visual features: (vision_embed_dim,)
        - Text features: (text_embed_dim,)
        - Weather: (weather_features,)
        - Component health: (num_components,)
        - Component age: (num_components,)
    
    Action Space (per agent):
        Discrete(4): 0=no action, 1=inspect, 2=preventive maintenance, 3=defer
    
    Reward:
        Multi-objective: minimize cost, maximize availability, early detection bonus
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, num_turbines: int = None, max_steps: int = None):
        """
        Initialize wind farm environment
        
        Args:
            num_turbines: Number of turbines (agents)
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.num_turbines = num_turbines or config.NUM_TURBINES
        self.max_steps = max_steps or config.MAX_STEPS_PER_EPISODE
        
        # Initialize data pipeline
        print("Initializing Wind Farm Environment...")
        self.data_pipeline = DataPipeline()
        
        # Initialize failure simulators for each turbine
        self.turbine_simulators = [
            TurbineFailureSimulator(i) for i in range(self.num_turbines)
        ]
        
        # Reward function
        self.reward_function = RewardFunction()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(config.NUM_DISCRETE_ACTIONS)
        
        # Observation space (multimodal)
        state_dims = self.data_pipeline.get_state_dimensions()
        
        self.observation_space = spaces.Dict({
            'scada': spaces.Box(
                low=-10, high=10,
                shape=state_dims['scada'],
                dtype=np.float32
            ),
            'vision': spaces.Box(
                low=-10, high=10,
                shape=state_dims['vision'],
                dtype=np.float32
            ),
            'text': spaces.Box(
                low=-10, high=10,
                shape=state_dims['text'],
                dtype=np.float32
            ),
            'weather': spaces.Box(
                low=-10, high=10,
                shape=state_dims['weather'],
                dtype=np.float32
            ),
            'component_health': spaces.Box(
                low=0, high=1,
                shape=(config.NUM_COMPONENTS_PER_TURBINE,),
                dtype=np.float32
            ),
            'component_age': spaces.Box(
                low=0, high=10000,
                shape=(config.NUM_COMPONENTS_PER_TURBINE,),
                dtype=np.float32
            )
        })
        
        # Episode tracking
        self.current_step = 0
        self.episode_stats = {
            'total_cost': 0,
            'total_energy_loss': 0,
            'num_failures': 0,
            'num_preventive_maintenance': 0,
            'total_downtime': 0
        }
        
        print(f"✓ Environment initialized with {self.num_turbines} turbines")
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            observations: Dict of observations for each agent
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_stats = {
            'total_cost': 0,
            'total_energy_loss': 0,
            'num_failures': 0,
            'num_preventive_maintenance': 0,
            'total_downtime': 0
        }
        
        # Reset all turbines
        for simulator in self.turbine_simulators:
            simulator.reset()
        
        # Get initial observations
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict[int, float], bool, bool, Dict]:
        """
        Execute actions for all agents
        
        Args:
            actions: Dict mapping agent_id to action
        
        Returns:
            observations: Next state observations
            rewards: Rewards for each agent
            terminated: Episode ended (failure/completion)
            truncated: Episode truncated (max steps)
            info: Additional information
        """
        self.current_step += 1
        
        # Store health before actions
        health_before = {}
        for agent_id in range(self.num_turbines):
            health_before[agent_id] = self.turbine_simulators[agent_id].get_component_health_vector()
        
        # Execute actions and calculate costs
        total_costs = {}
        total_energy_losses = {}
        
        for agent_id, action in actions.items():
            cost, energy_loss = self._execute_action(agent_id, action)
            total_costs[agent_id] = cost
            total_energy_losses[agent_id] = energy_loss
            
            self.episode_stats['total_cost'] += cost
            self.episode_stats['total_energy_loss'] += energy_loss
            
            if action == 2:  # Preventive maintenance
                self.episode_stats['num_preventive_maintenance'] += 1
        
        # Update turbine states with environmental factors
        env_factors = self._get_environmental_factors()
        
        for agent_id, simulator in enumerate(self.turbine_simulators):
            simulator.step(env_factors)
            
            # Check for failures
            if simulator.has_any_failure():
                self.episode_stats['num_failures'] += 1
                # Apply corrective maintenance (forced)
                for comp, failed in simulator.get_failure_status().items():
                    if failed:
                        simulator.apply_maintenance(comp, 'corrective')
                        # Add corrective costs
                        corr_cost = config.MAINTENANCE_COSTS['corrective'].get(comp, 50)
                        downtime = config.DOWNTIME['corrective'].get(comp, 48)
                        total_costs[agent_id] += corr_cost
                        energy_loss = self.reward_function.calculate_energy_loss(
                            downtime, env_factors.get('wind_speed', 10)
                        )
                        total_energy_losses[agent_id] += energy_loss
                        self.episode_stats['total_downtime'] += downtime
        
        # Get health after actions
        health_after = {}
        failures = {}
        for agent_id in range(self.num_turbines):
            health_after[agent_id] = self.turbine_simulators[agent_id].get_component_health_vector()
            failures[agent_id] = self.turbine_simulators[agent_id].get_failure_status()
        
        # Calculate rewards
        rewards = {}
        for agent_id in range(self.num_turbines):
            system_availability = 1.0 if not self.turbine_simulators[agent_id].has_any_failure() else 0.0
            
            reward, _ = self.reward_function.calculate_reward(
                action=actions.get(agent_id, 0),
                component_health_before=health_before[agent_id],
                component_health_after=health_after[agent_id],
                failures=failures[agent_id],
                maintenance_cost=total_costs[agent_id],
                energy_loss=total_energy_losses[agent_id],
                system_availability=system_availability
            )
            rewards[agent_id] = reward
        
        # Get new observations
        observations = self._get_observations()
        
        # Check termination conditions
        terminated = self.episode_stats['num_failures'] > self.num_turbines * 2  # Too many failures
        truncated = self.current_step >= self.max_steps
        
        # Info
        info = self._get_info()
        info['episode_stats'] = self.episode_stats.copy()
        
        return observations, rewards, terminated, truncated, info
    
    def _get_observations(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Get observations for all agents"""
        observations = {}
        
        for agent_id in range(self.num_turbines):
            # Get multimodal state from data pipeline
            state = self.data_pipeline.get_multimodal_state(
                turbine_id=agent_id,
                current_step=self.current_step
            )
            
            # Add component health and age from failure simulator
            simulator = self.turbine_simulators[agent_id]
            state['component_health'] = simulator.get_component_health_vector()
            state['component_age'] = simulator.get_component_age_vector()
            
            observations[agent_id] = state
        
        return observations
    
    def _get_environmental_factors(self) -> Dict[str, float]:
        """Get current environmental factors"""
        # Get weather from first turbine (simplified)
        weather = self.data_pipeline.weather_fetcher.get_weather('farm_1', self.current_step)
        
        factors = {
            'wind_speed': weather['current'][3] if len(weather['current']) > 3 else 10.0,
            'temperature': weather['current'][0] if len(weather['current']) > 0 else 15.0,
            'vibration': 0.5  # Could be extracted from SCADA
        }
        
        return factors
    
    def _execute_action(self, agent_id: int, action: int) -> Tuple[float, float]:
        """
        Execute maintenance action for an agent
        
        Args:
            agent_id: Turbine index
            action: Action to execute
        
        Returns:
            cost: Maintenance cost
            energy_loss: Energy production loss
        """
        simulator = self.turbine_simulators[agent_id]
        
        # Determine primary component to maintain (lowest health)
        component_health = simulator.get_component_health_vector()
        component_types = sorted(config.FAILURE_RATES.keys())
        weakest_component = component_types[np.argmin(component_health)]
        
        # Calculate cost
        cost = self.reward_function.calculate_action_cost(action, weakest_component)
        
        # Calculate downtime and energy loss
        env_factors = self._get_environmental_factors()
        downtime = self.reward_function.calculate_downtime_hours(action, weakest_component, False)
        energy_loss = self.reward_function.calculate_energy_loss(
            downtime, env_factors.get('wind_speed', 10)
        )
        
        # Apply maintenance if action is preventive (action=2)
        if action == 2:
            simulator.apply_maintenance(weakest_component, 'preventive')
            self.episode_stats['total_downtime'] += downtime
        
        return cost, energy_loss
    
    def _get_info(self) -> Dict:
        """Get additional environment information"""
        info = {
            'step': self.current_step,
            'system_health': [sim.get_system_health() for sim in self.turbine_simulators],
            'avg_system_health': np.mean([sim.get_system_health() for sim in self.turbine_simulators]),
            'num_turbines_operational': sum(1 for sim in self.turbine_simulators if not sim.has_any_failure())
        }
        return info
    
    def render(self):
        """Render environment state"""
        if self.current_step % 10 == 0:
            print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"System Health: {self._get_info()['avg_system_health']:.2f}")
            print(f"Total Cost: ${self.episode_stats['total_cost']:.1f}k")
            print(f"Failures: {self.episode_stats['num_failures']}")
            print(f"Preventive Maintenance: {self.episode_stats['num_preventive_maintenance']}")


# Test the environment
if __name__ == "__main__":
    print("Testing Wind Farm Environment...")
    print("=" * 60)
    
    # Create environment with fewer turbines for testing
    env = WindFarmEnv(num_turbines=5, max_steps=50)
    
    print(f"\nAction space: {env.action_space}")
    print(f"Observation space keys: {env.observation_space.spaces.keys()}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset(seed=42)
    print(f"Number of agents: {len(obs)}")
    print(f"Observation keys for agent 0: {obs[0].keys()}")
    print(f"Info: {info}")
    
    # Test step with random actions
    print("\nTesting random episode...")
    total_reward = {i: 0 for i in range(5)}
    
    for step in range(10):
        # Random actions for all agents
        actions = {i: env.action_space.sample() for i in range(5)}
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        for agent_id, reward in rewards.items():
            total_reward[agent_id] += reward
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Actions: {actions}")
            print(f"  Rewards: {list(rewards.values())[:3]}...")
            print(f"  Avg health: {info['avg_system_health']:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode finished!")
    print(f"Final stats: {info['episode_stats']}")
    print(f"Total rewards per agent: {list(total_reward.values())}")
    
    print("\n✓ Wind Farm Environment test passed!")
