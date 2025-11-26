"""
Experience Replay Buffer for Multi-Agent RL
Supports prioritized experience replay
"""

import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple

from config import config


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = None):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity or config.BUFFER_SIZE
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.use_per = config.USE_PER
        
        # PER parameters
        self.alpha = config.PER_ALPHA
        self.beta = config.PER_BETA_START
        self.beta_increment = (config.PER_BETA_END - config.PER_BETA_START) / config.TOTAL_TIMESTEPS
        self.epsilon = config.PER_EPS
    
    def push(self, states: Dict, actions: Dict, rewards: Dict, 
             next_states: Dict, done: bool, priority: float = None):
        """
        Add experience to buffer
        
        Args:
            states: Current states for all agents
            actions: Actions taken by all agents
            rewards: Rewards received by all agents
            next_states: Next states
            done: Episode done flag
            priority: Priority for PER (optional)
        """
        experience = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': done
        }
        
        self.buffer.append(experience)
        
        if self.use_per:
            if priority is None:
                # Max priority for new experiences
                max_priority = max(self.priorities) if self.priorities else 1.0
                self.priorities.append(max_priority)
            else:
                self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Sample batch of experiences
        
        Args:
            batch_size: Size of batch
        
        Returns:
            batch: Dict with batched experiences
            indices: Indices of sampled experiences
            weights: Importance sampling weights
        """
        if self.use_per:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)
    
    def _sample_uniform(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Uniform random sampling"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        # Convert to batched format
        batched = {
            'states': [exp['states'] for exp in batch],
            'actions': [exp['actions'] for exp in batch],
            'rewards': [exp['rewards'] for exp in batch],
            'next_states': [exp['next_states'] for exp in batch],
            'dones': [exp['dones'] for exp in batch]
        }
        
        weights = np.ones(batch_size)
        
        return batched, indices, weights
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Prioritized experience replay sampling"""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Increment beta
        self.beta = min(config.PER_BETA_END, self.beta + self.beta_increment)
        
        # Convert to batched format
        batched = {
            'states': [exp['states'] for exp in batch],
            'actions': [exp['actions'] for exp in batch],
            'rewards': [exp['rewards'] for exp in batch],
            'next_states': [exp['next_states'] for exp in batch],
            'dones': [exp['dones'] for exp in batch]
        }
        
        return batched, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        if not self.use_per:
            return
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


# Test replay buffer
if __name__ == "__main__":
    print("Testing Replay Buffer...")
    print("=" * 60)
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some experiences
    for i in range(100):
        states = {j: {'scada': np.random.randn(168, 13)} for j in range(5)}
        actions = {j: np.random.randint(4) for j in range(5)}
        rewards = {j: np.random.randn() for j in range(5)}
        next_states = {j: {'scada': np.random.randn(168, 13)} for j in range(5)}
        done = i % 50 == 49
        
        buffer.push(states, actions, rewards, next_states, done)
    
    print(f"\nBuffer size: {len(buffer)}")
    
    # Sample batch
    batch, indices, weights = buffer.sample(32)
    
    print(f"Batch size: {len(batch['states'])}")
    print(f"Indices shape: {indices.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Sample weights: {weights[:5]}")
    
    print("\n✓ Replay Buffer test passed!")
