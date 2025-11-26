"""
Multi-Agent Reinforcement Learning Agents
Implements QMIX algorithm with communication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List

from config import config
from encoders import MultimodalEncoder


class QNetwork(nn.Module):
    """Individual Q-network for each agent"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, config.Q_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(config.Q_HIDDEN_DIMS[0], config.Q_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(config.Q_HIDDEN_DIMS[1], config.Q_HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Linear(config.Q_HIDDEN_DIMS[2], action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class CommunicationModule(nn.Module):
    """Agent communication using learnable message passing"""
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        # Message generation
        self.message_gen = nn.Sequential(
            nn.Linear(state_dim, config.COMM_DIM),
            nn.ReLU(),
            nn.Linear(config.COMM_DIM, config.COMM_DIM)
        )
        
        # Message aggregation
        self.message_agg = nn.Sequential(
            nn.Linear(config.COMM_DIM, config.COMM_DIM),
            nn.ReLU(),
            nn.Linear(config.COMM_DIM, state_dim)
        )
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Communication round
        
        Args:
            states: States of all agents (batch, num_agents, state_dim)
        
        Returns:
            Updated states after communication (batch, num_agents, state_dim)
        """
        batch_size, num_agents, state_dim = states.shape
        
        # Generate messages from each agent
        messages = self.message_gen(states)  # (batch, num_agents, comm_dim)
        
        # Aggregate messages (simple averaging, could use attention)
        aggregated = messages.mean(dim=1, keepdim=True)  # (batch, 1, comm_dim)
        aggregated = aggregated.expand(-1, num_agents, -1)  # Broadcast to all agents
        
        # Process aggregated messages
        comm_features = self.message_agg(aggregated)  # (batch, num_agents, state_dim)
        
        # Add to original states (residual)
        return states + comm_features


class MixingNetwork(nn.Module):
    """QMIX mixing network for credit assignment"""
    
    def __init__(self, num_agents: int, state_dim: int):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = config.MIXING_EMBED_DIM
        
        # Hypernetworks for generating mixing network weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, config.MIXING_HYPERNET_HIDDEN),
            nn.ReLU(),
            nn.Linear(config.MIXING_HYPERNET_HIDDEN, num_agents * self.embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, config.MIXING_HYPERNET_HIDDEN),
            nn.ReLU(),
            nn.Linear(config.MIXING_HYPERNET_HIDDEN, self.embed_dim)
        )
        
        # Hypernetworks for biases
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mix agent Q-values into total Q-value
        
        Args:
            agent_qs: Individual Q-values (batch, num_agents)
            states: Global state (batch, state_dim)
        
        Returns:
            Mixed Q-value (batch, 1)
        """
        batch_size = agent_qs.size(0)
        
        # Generate mixing network weights from state
        w1 = torch.abs(self.hyper_w1(states))  # Ensure monotonicity
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)
        
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        
        # First layer
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        
        # Final Q-value
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.view(batch_size, 1)


class MARLAgent:
    """Multi-Agent RL Agent with QMIX"""
    
    def __init__(self, num_agents: int, state_dim: int, action_dim: int):
        """
        Initialize MARL agent
        
        Args:
            num_agents: Number of agents
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Multimodal encoder (shared)
        self.encoder = MultimodalEncoder().to(self.device)
        
        # Q-networks for each agent
        self.q_networks = nn.ModuleList([
            QNetwork(config.FINAL_STATE_DIM, action_dim).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Target networks
        self.target_encoder = MultimodalEncoder().to(self.device)
        self.target_q_networks = nn.ModuleList([
            QNetwork(config.FINAL_STATE_DIM, action_dim).to(self.device)
            for _ in range(num_agents)
        ])
        
        # Copy weights to target networks
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for i in range(num_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        # Communication module
        if config.USE_COMMUNICATION:
            self.comm_module = CommunicationModule(config.FINAL_STATE_DIM).to(self.device)
        
        # Mixing network
        self.mixer = MixingNetwork(num_agents, config.FINAL_STATE_DIM).to(self.device)
        self.target_mixer = MixingNetwork(num_agents, config.FINAL_STATE_DIM).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.q_networks.parameters()) +
            list(self.mixer.parameters()) +
            (list(self.comm_module.parameters()) if config.USE_COMMUNICATION else []),
            lr=config.LEARNING_RATE
        )
        
        # Exploration
        self.epsilon = config.EPS_START
    
    def select_actions(self, states: Dict[int, Dict], explore: bool = True) -> Dict[int, int]:
        """
        Select actions for all agents
        
        Args:
            states: Dict of states for each agent
            explore: Whether to explore (epsilon-greedy)
        
        Returns:
            Dict of actions for each agent
        """
        actions = {}
        
        # Convert states to tensors
        batch_states = self._prepare_batch_states(states)
        
        with torch.no_grad():
            # Encode states
            encoded_states = []
            for agent_id in range(self.num_agents):
                agent_state = {k: v[agent_id:agent_id+1] for k, v in batch_states.items()}
                encoded = self.encoder(agent_state)
                encoded_states.append(encoded)
            
            encoded_states = torch.cat(encoded_states, dim=0)  # (num_agents, state_dim)
            
            # Communication
            if config.USE_COMMUNICATION:
                encoded_states = self.comm_module(encoded_states.unsqueeze(0)).squeeze(0)
            
            # Get Q-values and select actions
            for agent_id in range(self.num_agents):
                q_values = self.q_networks[agent_id](encoded_states[agent_id])
                
                if explore and np.random.rand() < self.epsilon:
                    actions[agent_id] = np.random.randint(self.action_dim)
                else:
                    actions[agent_id] = q_values.argmax().item()
        
        return actions
    
    def train_step(self, batch: Dict) -> float:
        """
        Training step using QMIX
        
        Args:
            batch: Batch of experiences
        
        Returns:
            Loss value
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        batch_size = len(states)
        
        # Encode current states for all agents
        current_encoded = []
        for i in range(batch_size):
            agent_encoded = []
            for agent_id in range(self.num_agents):
                # Get single agent's state from the dict
                single_agent_state = {}
                for k, v in states[i][agent_id].items():
                    # Convert to tensor and add batch dimension
                    single_agent_state[k] = torch.FloatTensor(v).unsqueeze(0).to(self.device)
                
                encoded = self.encoder(single_agent_state)
                agent_encoded.append(encoded)
            current_encoded.append(torch.cat(agent_encoded, dim=0))
        
        current_encoded = torch.stack(current_encoded)  # (batch, num_agents, state_dim)
        
        # Communication
        if config.USE_COMMUNICATION:
            current_encoded = self.comm_module(current_encoded)
        
        # Get Q-values for taken actions
        chosen_action_qvals = []
        for agent_id in range(self.num_agents):
            q_vals = self.q_networks[agent_id](current_encoded[:, agent_id])
            agent_actions = torch.LongTensor([actions[i][agent_id] for i in range(batch_size)]).to(self.device)
            chosen_q = q_vals.gather(1, agent_actions.unsqueeze(1))
            chosen_action_qvals.append(chosen_q)
        
        chosen_action_qvals = torch.cat(chosen_action_qvals, dim=1)  # (batch, num_agents)
        
        # Mix Q-values
        global_state = current_encoded.mean(dim=1)  # Simple aggregation
        q_total = self.mixer(chosen_action_qvals, global_state)
        
        # Target Q-values
        with torch.no_grad():
            next_encoded = []
            for i in range(batch_size):
                agent_encoded = []
                for agent_id in range(self.num_agents):
                    single_agent_state = {}
                    for k, v in next_states[i][agent_id].items():
                        single_agent_state[k] = torch.FloatTensor(v).unsqueeze(0).to(self.device)
                    
                    encoded = self.target_encoder(single_agent_state)
                    agent_encoded.append(encoded)
                next_encoded.append(torch.cat(agent_encoded, dim=0))
            
            next_encoded = torch.stack(next_encoded)
            
            if config.USE_COMMUNICATION:
                next_encoded = self.comm_module(next_encoded)
            
            target_max_qvals = []
            for agent_id in range(self.num_agents):
                target_q = self.target_q_networks[agent_id](next_encoded[:, agent_id])
                target_max_qvals.append(target_q.max(1, keepdim=True)[0])
            
            target_max_qvals = torch.cat(target_max_qvals, dim=1)
            
            next_global_state = next_encoded.mean(dim=1)
            target_q_total = self.target_mixer(target_max_qvals, next_global_state)
            
            # Compute targets
            rewards_tensor = torch.FloatTensor([sum(rewards[i].values()) for i in range(batch_size)]).to(self.device)
            dones_tensor = torch.FloatTensor([float(dones[i]) for i in range(batch_size)]).to(self.device)
            
            targets = rewards_tensor.unsqueeze(1) + config.GAMMA * target_q_total * (1 - dones_tensor.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(q_total, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.MAX_GRAD_NORM)
        self.optimizer.step()
        
        return loss.item()
    
    def parameters(self):
        """Get all trainable parameters"""
        params = (list(self.encoder.parameters()) +
                 list(self.q_networks.parameters()) +
                 list(self.mixer.parameters()))
        if config.USE_COMMUNICATION:
            params += list(self.comm_module.parameters())
        return params
    
    def update_target_networks(self):
        """Soft update of target networks"""
        # Encoder
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)
        
        # Q-networks
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_q_networks[i].parameters(), 
                                          self.q_networks[i].parameters()):
                target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)
        
        # Mixer
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(config.TAU * param.data + (1 - config.TAU) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(config.EPS_END, self.epsilon * config.EPS_DECAY)
    
    def _prepare_batch_states(self, states: Dict[int, Dict]) -> Dict[str, torch.Tensor]:
        """Convert dict of states to batched tensors"""
        batch_states = {}
        
        # Get all modality keys from first agent
        first_agent = list(states.keys())[0]
        modalities = list(states[first_agent].keys())
        
        # Stack each modality across agents
        for modality in modalities:
            batch_states[modality] = torch.stack([
                torch.FloatTensor(states[agent_id][modality]) 
                for agent_id in sorted(states.keys())
            ]).to(self.device)
        
        return batch_states
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'q_networks': [q.state_dict() for q in self.q_networks],
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        for i, q_dict in enumerate(checkpoint['q_networks']):
            self.q_networks[i].load_state_dict(q_dict)
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# Test MARL agent
if __name__ == "__main__":
    print("Testing MARL Agent...")
    print("=" * 60)
    
    num_agents = 5
    state_dim = config.FINAL_STATE_DIM
    action_dim = config.NUM_DISCRETE_ACTIONS
    
    agent = MARLAgent(num_agents, state_dim, action_dim)
    
    print(f"\nAgent initialized:")
    print(f"  Num agents: {num_agents}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Epsilon: {agent.epsilon:.3f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    print("\n✓ MARL Agent test passed!")
